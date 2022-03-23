
struct TraceSim <: MethodType
    α::Float64
    β::Float64
    γ::Float64
    
    sigmoid::Bool
    b::Float64 
    
    idf::Bool
    reciprocal_func::Bool
    no_norm::Bool

    sum_match::Bool


    local_weights::Vector{Float64}
end


function TraceSim(α::Float64, β::Float64, γ::Float64, sigmoid::Bool, b::Float64, idf::Bool, reciprocal_func::Bool,  no_norm::Bool, sum_match::Bool=false)
    local_weights = Vector{Float64}(undef, 400)

    @simd for pos in 1:400
        @inbounds local_weights[pos] = reciprocal_func ? 1.0 / ((pos)^α) : exp(-α * (pos - 1.0))
    end

    @info "TraceSim: sum_match=$(sum_match)"
    return TraceSim(α, β, γ, sigmoid, b, idf, reciprocal_func, no_norm, sum_match, local_weights)
end

computeDocFreq(method::TraceSim, docFreq) = @. max(exp(-method.β * (docFreq.doc_freq / docFreq.nDocs) * 100.0), 1.0e-100)


@inline function frame_weight(method::TraceSim, pos, df)
    @inbounds lw = method.local_weights[pos]

    return lw * df
end

gap_score(method::TraceSim, pos, df) = -1.0 * frame_weight(method, pos, df)
@inline function match_score(method::TraceSim, pos1, pos2, df, qw, cw)
    diff = exp(-method.γ * abs(pos1 - pos2))

    if method.sum_match
        return (qw + cw) * diff
    else
        return max(qw, cw) * diff
    end
end


function compute_frame_weights(method::TraceSim, stacktrace::Vector{UInt32}, doc_freq::Vector{Float64})
    weights = Vector{Float64}(undef, length(stacktrace))
    
    @simd for pos in eachindex(stacktrace)
        @inbounds weights[pos] = frame_weight(method, pos, doc_freq[stacktrace[pos]])
    end

    return weights
end

@inline norm_sum(sim, q_weights::Vector{Float64}, c_weights::Vector{Float64})= sim/(sum(q_weights) + sum(c_weights))

function norm_jaccard(sim, query::Vector{UInt32}, candidate::Vector{UInt32}, q_weights::Vector{Float64}, c_weights::Vector{Float64},q_sorted_idxs, c_sorted_idxs)
    i = 1
    j = 1
    den = 0.0

    query_len = length(query) + 1
    cand_len = length(candidate) + 1

    func_id = 0x00000

    @inbounds while i < query_len || j < cand_len
        aux1 = 0.0
        
        if i == query_len || (j < cand_len && query[q_sorted_idxs[i]] > candidate[c_sorted_idxs[j]])
            # Query does not contain the candidate[c_sorted_idxs[j]]
            func_id = candidate[c_sorted_idxs[j]]

            while j < cand_len
                candidate[c_sorted_idxs[j]] != func_id && break

                aux1 += c_weights[c_sorted_idxs[j]]
                j += 1
            end
            
            den += aux1
        elseif j == cand_len || ( i < query_len && query[q_sorted_idxs[i]] < candidate[c_sorted_idxs[j]])
            # Candidate does not contain the query[q_sorted_idxs[i]]
            func_id = query[q_sorted_idxs[i]]

            while i < query_len
                query[q_sorted_idxs[i]] != func_id && break
                
                aux1 += q_weights[q_sorted_idxs[i]]
                i += 1
            end

            den += aux1
        else
            # Both sequences contain a function x. Get maximum weights between them
            
            # Sum of the weights of a function in candidate
            func_id = candidate[ c_sorted_idxs[j]]

            while j < cand_len
                candidate[ c_sorted_idxs[j]] != func_id && break
                aux1 += c_weights[ c_sorted_idxs[j]]
                j += 1
            end

            # Sum of the weights of a function in query
            aux2 = 0.0
            func_id = query[q_sorted_idxs[i]]

            while i < query_len
                query[q_sorted_idxs[i]] != func_id && break

                aux2 += q_weights[q_sorted_idxs[i]]
                i += 1
            end

            den += max(aux1, aux2)
        end
    end

    den == 0.0 && return -1.0

    return sim / den
end


@inline function similarity(method::TraceSim, query::Vector{UInt32}, candidate::Vector{UInt32}, doc_freq::Vector{Float64})
    q_weights = compute_frame_weights(method, query, doc_freq)
    
    return similarity(method, query, candidate, q_weights, sortperm(query), sortperm(candidate), doc_freq)
end


function similarity(method::TraceSim, query::Vector{UInt32}, candidate::Vector{UInt32}, q_weights::Vector{Float64}, q_sorted_idxs, c_sorted_idxs, doc_freq::Vector{Float64})
    c_weights = compute_frame_weights(method, candidate, doc_freq)

    query_len = length(q_weights) 
    cand_len = length(c_weights)
    offset = cand_len + 1

    # previous_row = zeros(offset) 
    row = zeros(offset)

    # Create initial values
    for i in 1:cand_len
        @inbounds row[i + 1] =  row[i] - c_weights[i]
    end


    @inbounds for (q_pos, q_func) in enumerate(query)
        # Set first column of the row
        diagonal = row[1]
        row[1] -= q_weights[q_pos]
        
        for (c_pos, c_func) in enumerate(candidate)
            col_idx = c_pos + 1
            
            # Align gap to query position  
            above = row[col_idx] - q_weights[q_pos]
            # println("\tPrevious row= $(previous_row[col_idx]) - $(q_weights[q_pos])")
            
            # Align gap to candidate position
            left = row[col_idx - 1] - c_weights[c_pos]
            # println("\tPrevious col= $(cur_row[col_idx - 1]) - $(c_weights[c_pos])")

            # Value of diagonal
            # diagonal = previous_row[col_idx - 1]

            if q_func == c_func
                # Matching
                diagonal += match_score(method, q_pos, c_pos, 0.0, q_weights[q_pos], c_weights[c_pos])
            else
                # Mismatch
                diagonal -= q_weights[q_pos] + c_weights[c_pos]
                # println("\tDiagonal = $(previous_row[col_idx - 1]) - $( q_weights[q_pos] + c_weights[c_pos])")
            end
            
            diagonal_tmp = row[col_idx]

            # Replace max with IF ELSE considerably improves the performance =O
            # row[col_idx] = max(left, above, diagonal)
            m = left

            if above > m
                m = above
            end

            if diagonal > m
                m = diagonal
            end

            row[col_idx] = m

            diagonal = diagonal_tmp
            # println("$(q_pos), $(c_pos) max($(left), $(diagonal_value), $(above))")      
        end
    end
        
    sim = row[offset]

    if !method.no_norm
        if method.sum_match
            return norm_sum(sim, q_weights, c_weights)
        else
            return norm_jaccard(sim, query, candidate, q_weights, c_weights, q_sorted_idxs, c_sorted_idxs)
        end
    end

    return sim
end


# @inline function compute_final_similarity!(partial_similarities::Vector{Float64}, method::TraceSim, query::Report, candidates::Vector{Report}, df_vec::Vector{Float64})
#     query_weight_sum = 0.0
#     q_weights = Vector{Float64}(undef, query.stats.doc_len)
#     q_frames = Vector{UInt32}(undef, query.stats.doc_len)

#     stack_idx = 1

#     for stack in query.stacks
#         for (pos, func_id) in enumerate(stack)
#             fw = frame_weight(method, pos, df_vec[func_id])
            
#             q_weights[stack_idx] = fw
#             q_frames[stack_idx] = func_id

#             query_weight_sum -= fw 
#             stack_idx+=1
#         end
#     end

        
#     Threads.@threads for idx in 1:length(partial_similarities)
#         sim = partial_similarities[idx]

#         if sim == 0.0
#             partial_similarities[idx] = -1.0
#         else
#             weight_sum = query_weight_sum
#             @inbounds candidate = candidates[idx]

#             c_weights = Vector{Float64}(undef, candidate.stats.doc_len)
#             c_frames = Vector{UInt32}(undef, candidate.stats.doc_len)

#             c_stack_idx = 1

#             for stack in candidate.stacks
#                 for (pos, func_id) in enumerate(stack)
#                     @inbounds fw = frame_weight(method, pos, df_vec[func_id])
#                     c_weights[c_stack_idx] = fw
#                     c_frames[c_stack_idx] = func_id

#                     weight_sum -= fw
#                     c_stack_idx+=1
#                 end
#             end

#             sim += weight_sum  

#             partial_similarities[idx] = normalize_sim(sim, q_frames, c_frames, q_weights, c_weights, sortperm(q_frames), sortperm(c_frames))
#         end   
#     end

#     return partial_similarities
# end


# function similarity(method::TraceSim, query::Vector{UInt32}, candidate::Vector{UInt32}, doc_freq::Vector{Float64})
#     q_weights = compute_frame_weights(method, query, doc_freq)
#     c_weights = compute_frame_weights(method, candidate,doc_freq)

#     offset = length(candidate) + 1
#     M = zeros(2 * offset)

#     query_len = length(q_weights) 
#     cand_len = length(c_weights)

#     ONE_AGO = 0
#     THIS_ROW = 1

#     # Create initial values
#     for i in 1:cand_len
#         M[THIS_ROW * offset + (i + 1)] =  M[THIS_ROW * offset + i] - c_weights[i]
#     end
    
#     # println(M[THIS_ROW * offset + 1 : THIS_ROW * offset + offset])
#     for (q_pos, q_func) in enumerate(query)
#         # # Copy THIS_ROW to ONE_AGO
#         for k in 1:offset
#             M[ONE_AGO * offset + k] = M[THIS_ROW * offset + k]
#         end

#         # Reset THIS_ROW
#         for k in 1:offset
#             M[THIS_ROW * offset + k] = 0.0
#         end

#         # Set first column of the row
#         M[THIS_ROW * offset + 1] = M[ONE_AGO * offset + 1] - q_weights[q_pos]
        

#         for (c_pos, c_func) in enumerate(candidate)
#             col_idx = c_pos +1
            
#             # Align gap to query position  
#             previous_row = M[ONE_AGO * offset + col_idx] - q_weights[q_pos]
#             # println("\tPrevious row= $(M[ONE_AGO * offset + col_idx]) - $(q_weights[q_pos])")
            
#             # Align gap to candidate position
#             previous_col = M[THIS_ROW * offset + col_idx - 1] - c_weights[c_pos]
#             # println("\tPrevious col= $(M[THIS_ROW * offset + col_idx - 1]) - $(c_weights[c_pos])")

#             # Value of diagonal
#             diagonal_value = M[ONE_AGO * offset + col_idx - 1]

            
#             if q_func == c_func
#                 #Matching
#                 diagonal_value += max(q_weights[q_pos], c_weights[c_pos]) * exp(-method.γ * abs(q_pos - c_pos))
#             else
#                 #Mismatch
#                 diagonal_value -= q_weights[q_pos] + c_weights[c_pos]
#                 # println("\tDiagonal = $(M[ONE_AGO * offset + col_idx - 1]) - $( q_weights[q_pos] + c_weights[c_pos])")
#             end

#             M[THIS_ROW * offset + col_idx] = max(previous_row, previous_col, diagonal_value)
#             # println("$(q_pos), $(c_pos) max($(previous_row), $(diagonal_value), $(previous_col))")
#         end
#         # println(M[THIS_ROW * offset + 1: THIS_ROW * offset + offset])
#     end
    
#     sim = M[THIS_ROW * offset + offset]

#     if !method.no_norm
#         return normalize(sim, query, candidate, q_weights, c_weights)
#     end

#     return sim
# end