module Method

using util

export computeDocFreq, MethodType, TraceSim, parseMethod, createMethod, compute_frame_weights, similarity, normalizeDocFreq, 
    MlModel, compute_frame_weight, match_score, update_method_param!, compute_final_similarity!, gap_score,
    TFIDF, BM25, compute_tf, PDM, IRMethod, FaST, frame_weight, CosineTFIDF, compute_term_sim, PrefixMatch, normalize_vec


abstract type MethodType end

normalizeDocFreq(docFreq::DocFreq)::Vector{Float64}= @. (docFreq.doc_freq / docFreq.nDocs) * 100.0


function computeDocFreq(method::MethodType, docFreq)::Vector{Float64}
    return Vector{Float64}()
end


include("methods/trace_sim.jl")
# include("methods/brodie05.jl")
include("methods/pdm.jl")
include("methods/prefix_match.jl")
include("methods/IR_method.jl")
include("methods/fast.jl")


end