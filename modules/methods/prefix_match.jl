
struct PrefixMatch <: MethodType
end

computeDocFreq(method::PrefixMatch, docFreq) = docFreq.doc_freq
   
function similarity(method::PrefixMatch, query::Vector{UInt32}, candidate::Vector{UInt32}, doc_freq::Vector{Float64})
    q_len = length(query)
    c_len = length(candidate)
    prefix_len = 0.0

    for i=1:min(q_len, c_len)
        @inbounds query[i] != candidate[i] && break
        prefix_len += 1.0
    end

    return prefix_len / max(q_len, c_len)
end