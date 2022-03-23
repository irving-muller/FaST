
struct PDM <: MethodType
    c::Float64
    o::Float64
    norms::Vector{Float64}
end

@inline function PDM(c::Float64, o::Float64)
    norms = Vector{Float64}(undef,12000)
    previous = 0.0

    for cur_pos in 0:11999
        previous = cur_pos == 0 ? 0.0 : norms[cur_pos]
        norms[cur_pos + 1] = previous + exp(-c * cur_pos)
    end

    return PDM(c, o, norms)
end


computeDocFreq(method::PDM, docFreq) = docFreq.doc_freq

gap_score(method::PDM, pos, ndf) = 0.0
match_score(method::PDM, q_pos, c_pos, ndf, q_gap_score, c_gap_score) =  exp(-method.c * (min(q_pos, c_pos) - 1.0)) * exp(-method.o * abs(q_pos - c_pos))

# @inline function compute_final_similarity!(similarities, method::PDM, query, candidates, df_vec)
#     query_len = query.stats.doc_len

#     @simd for i = eachindex(similarities)
#         norm = method.norms[min(query_len, candidates[i].stats.doc_len)]
#         @inbounds similarities[i] /= norm
#     end

#     return similarities
# end

   
function similarity(method::PDM, query::Vector{UInt32}, candidate::Vector{UInt32}, doc_freq::Vector{Float64})
    q_len = length(query) 
    c_len = length(candidate)

    row = zeros(c_len)
    
    @inbounds for (q_pos, q_func) in enumerate(query)
        diagonal = 0.0
        
        for (c_pos, c_func) in enumerate(candidate)            
            # Align gap to query position  
            above = row[c_pos]
            
            # Align gap to candidate position
            left = c_pos == 1 ? 0.0 : row[c_pos - 1]

            if q_func == c_func
                # Matching
                diagonal += match_score(method, q_pos, c_pos, 0.0, 0.0, 0.0)
            end
            
            diagonal_tmp = row[c_pos]


        
            # Replace max with IF ELSE considerably improves the performance =O
            # row[c_pos] = max(left, above, diagonal)
            m = left

            if above > m
                m = above
            end

            if diagonal > m
                m = diagonal
            end

            row[c_pos] = m
            
            
            diagonal = diagonal_tmp
        end
    end

    @inbounds sim = row[c_len]
    norm = method.norms[min(q_len, c_len)]
    
    return sim / norm
end