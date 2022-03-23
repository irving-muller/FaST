
struct FaST    <: MethodType
    α::Float64
    β::Float64
    γ::Float64

    local_weights::Vector{Float64}
    compute_weight_sum::Bool
end

function FaST(α::Float64, β::Float64, γ::Float64, compute_weight_sum::Bool)
    local_weights = Vector{Float64}(undef, 400)

    @simd for pos in 1:400
        @inbounds local_weights[pos] = max(pos^-α, 1.0e-100)
    end

    return FaST(α, β, γ, local_weights, compute_weight_sum)
end

@inline computeDocFreq(method::FaST, docFreq) = @. max(exp(-method.β * (docFreq.doc_freq / docFreq.nDocs) * 100.0), 1.0e-100)

@inline function frame_weight(method::FaST, pos, doc_freq)    
    @inbounds return method.local_weights[pos] * doc_freq
end

gap_score(method::FaST, pos, doc_freq) = -1.0 * frame_weight(method, pos, doc_freq)
match_score(method::FaST, pos1, pos2, doc_freq, qw, cw) = (qw + cw) * exp(-method.γ * abs(pos1 - pos2))




@inline function sum_weight_positions(method::FaST, positions, start_idx, doc_freq)
    sum_fw = 0.0

    @inbounds for idx=start_idx:length(positions)
        sum_fw += frame_weight(method, positions[idx], doc_freq)
    end

    return sum_fw
end


function similarity(method::FaST, query::Vector{Pair{UInt32, Vector{Int16}}}, candidate::Vector{Pair{UInt32, Vector{Int16}}}, df_vec::Vector{Float64})
    q_idx = 1
    c_idx = 1

    sim = 0.0
    den = 0.0

    q_len = length(query) + 1
    c_len = length(candidate) + 1

    @inbounds while q_idx != q_len && c_idx != c_len
        q_func_id, q_positions = query[q_idx]
        c_func_id, c_positions = candidate[c_idx]

        if q_func_id == c_func_id
            # Query and candidate contain the same function. Align positions starting from the left
            doc_freq = df_vec[q_func_id]

            if length(q_positions) < length(c_positions)
                smallest_vec = q_positions
                longest_vec = c_positions
            else
                smallest_vec = c_positions
                longest_vec = q_positions
            end

            for j=1:length(smallest_vec)
                q_pos = q_positions[j]
                c_pos = c_positions[j]

                q_fw = frame_weight(method, q_pos, doc_freq)
                c_fw = frame_weight(method, c_pos, doc_freq)
                
                sim +=  match_score(method::FaST, q_pos, c_pos, doc_freq, q_fw, c_fw)
                den += (q_fw + c_fw)
                j+=1
            end

            sum_fw = sum_weight_positions(method, longest_vec, length(smallest_vec) + 1, doc_freq)
            sim -= sum_fw
            den += sum_fw

            q_idx +=1
            c_idx +=1
        else
            if q_func_id > c_func_id
                # c_func_id does not exist in the query
                cur_positions = c_positions
                doc_freq = df_vec[c_func_id]
                c_idx += 1
            else
                # q_func_id does not exist in the candidate
                cur_positions = q_positions
                doc_freq = df_vec[q_func_id]
                q_idx += 1
            end

            # Align to gaps to all positions
            sum_fw = sum_weight_positions(method, cur_positions, 1, doc_freq)
            sim -= sum_fw
            den += sum_fw
        end
    end

    
    @inbounds for j=q_idx:length(query)
        func_id, positions = query[j]
        doc_freq = df_vec[func_id]

        sum_fw = sum_weight_positions(method, positions, 1, doc_freq)
        sim -= sum_fw
        den += sum_fw
    end

    @inbounds for j=c_idx:length(candidate)
        func_id, positions = candidate[j]
        doc_freq = df_vec[func_id]

        sum_fw = sum_weight_positions(method, positions, 1, doc_freq)
        sim -= sum_fw
        den += sum_fw
    end

    return sim/den    
end


@inline function sum_frame_weights(method::FaST, report::Report, df_vec::Vector{Float64})
    sum = 0.0

    for stack in report.stacks 
        for (pos, func_id) in enumerate(stack)
            @inbounds sum += frame_weight(method, pos, df_vec[func_id])
        end
    end

    return sum
end


@inline function compute_final_similarity!(partial_similarities::Vector{Float64}, method::FaST, query::Report, candidates::Vector{Report}, df_vec::Vector{Float64})
    if method.compute_weight_sum
        q_weight_sum = sum_frame_weights(method, query, df_vec)
        
        Threads.@threads for idx in eachindex(partial_similarities)
            @inbounds sim = partial_similarities[idx]
            
            if sim == 0.0        
                @inbounds partial_similarities[idx] = -1.0
            else
                @inbounds weight_sum = q_weight_sum + sum_frame_weights(method, candidates[idx], df_vec)

                # Correct similarities and then normalize it
                @inbounds partial_similarities[idx] = (sim - weight_sum) / weight_sum
            end   
        end
    else 
        q_weight_sum = query.stats.weight_sum

        for idx in eachindex(partial_similarities)
            @inbounds sim = partial_similarities[idx]
            
            if sim == 0.0        
                @inbounds partial_similarities[idx] = -1.0
            else
                @inbounds weight_sum = q_weight_sum + candidates[idx].stats.weight_sum

                # Correct similarities and then normalize it
                @inbounds partial_similarities[idx] = (sim - weight_sum) / weight_sum
            end   
        end
    end

    return partial_similarities
end