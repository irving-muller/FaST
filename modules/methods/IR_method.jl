abstract type IRMethod <: MethodType end


####################################################################
# BM25
####################################################################

mutable struct BM25 <: IRMethod
    k1::Float64
    b::Float64
end

computeDocFreq(method::BM25, docFreq::DocFreq) = @. log(1.0 + (docFreq.nDocs - docFreq.doc_freq + 0.5) / (docFreq.doc_freq + 0.5))
compute_tf(method::BM25, tf, doc_len, doc_avg) = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * doc_len / doc_avg))
compute_term_sim(method::BM25, term_idf, q_weight, c_weight, doc_len, doc_avg) = compute_tf(method::BM25, c_weight, doc_len, doc_avg) * term_idf
generate_bow_vector(method::BM25, report::Report, df_vec::Vector{Float64}, vocab::Vocab, ngram::Int64) = transform_BOW(report, vocab::Vocab, ngram::Int64)

normalize_vec(method::BM25, bow::Vector{Pair{UInt32,Float64}}, df_vec::Vector{Float64})= bow



compute_final_similarity!(partial_similarities::Vector{Float64}, method::BM25, query::Report, candidates::Vector{Report}, df_vec::Vector{Float64}) = partial_similarities

####################################################################
# TF-IDF
####################################################################


@enum tf_scheme TF_RAW = 1  TF_SQRT = 2 TF_LOG = 3
@enum idf_scheme IDF_STD = 1 IDF_SMOOTH = 2 IDF_SMOOTH_PLUS = 3

function get_tf_scheme(tf_scheme_nm::String)
    if tf_scheme_nm == "raw"
        return TF_RAW
    elseif tf_scheme_nm == "log"
        return TF_LOG
    elseif tf_scheme_nm == "sqrt"
        return TF_SQRT
    end

    return TF_RAW
end 

function get_idf_scheme(idf_scheme_nm::String)
    if idf_scheme_nm == "std"
        return IDF_STD
    elseif idf_scheme_nm == "smooth"
        return IDF_SMOOTH
    elseif idf_scheme_nm == "smooth_plus"
        return IDF_SMOOTH_PLUS
    end

    return IDF_SMOOTH
end 

mutable struct TFIDF <: IRMethod
    tf_scheme::tf_scheme
    idf_scheme::idf_scheme
    len_norm::Vector{Float64}
    without_len_norm::Bool
    add_query_freq::Bool
end

function TFIDF(tf_scheme_nm::String, idf_scheme_nm::String, without_len_norm=false, add_query_freq=false)
    tf_scheme = get_tf_scheme(tf_scheme_nm)
    idf_scheme = get_idf_scheme(idf_scheme_nm)

    @info("TF: $(tf_scheme); IDF: $(idf_scheme)")

    if without_len_norm
        len_norm = [1.0]
    else
        len_norm = Vector{Float64}(undef, 12000)

        for doc_len in 1:12000
            len_norm[doc_len] = 1.0 / sqrt(doc_len)
        end

    end

    return TFIDF(tf_scheme, idf_scheme, len_norm, without_len_norm, add_query_freq)
end

tf_sqrt(tf) = sqrt(tf)
tf_log(tf) = log(1 + tf)

idf_std(docFreq::DocFreq) = @. log(docFreq.nDocs / docFreq.doc_freq)
idf_smooth(docFreq::DocFreq) = @. log((docFreq.nDocs + 1.0) / (docFreq.doc_freq + 1.0))
idf_smooth_plus(docFreq::DocFreq) = idf_smooth(docFreq) .+ 1.0


@inline function computeDocFreq(method::TFIDF, docFreq::DocFreq)
    if method.idf_scheme == IDF_SMOOTH
        return idf_smooth(docFreq)
    elseif method.idf_scheme == IDF_STD
        return idf_std(docFreq)
    elseif method.idf_scheme == IDF_SMOOTH_PLUS
        return idf_smooth_plus(docFreq)
    else
        return idf_smooth(docFreq)
    end
end

@inline function compute_tf(method::TFIDF, raw_tf, doc_len, doc_avg)
    len_norm = method.without_len_norm ? method.len_norm[1] : method.len_norm[doc_len]
        
    if method.tf_scheme == TF_SQRT
        return len_norm * tf_sqrt(raw_tf)
    elseif method.tf_scheme == TF_LOG
        return len_norm * tf_log(raw_tf)
    else    
        return len_norm * raw_tf
    end
end

function similarity(method::TFIDF, q_bow, c_bow, idf_vec, cand_len)
    q_idx = 1
    c_idx = 1
    sim = 0.0

    q_len = length(q_bow) + 1
    c_len = length(c_bow) + 1

    @inbounds while q_idx != q_len && c_idx != c_len
        q_func_id, q_tf = q_bow[q_idx]
        c_func_id, c_tf = c_bow[c_idx]

        if q_func_id == c_func_id
            # Query and candidate contain the same function. Align positions starting from the left
            term_idf = idf_vec[q_func_id]
            sim += compute_term_sim(method, term_idf, q_tf, c_tf, cand_len, -0.0)

            q_idx +=1
            c_idx +=1
        elseif q_func_id > c_func_id
            # c_func_id does not exist in the query
            c_idx += 1
        else
            # q_func_id does not exist in the candidate
            q_idx += 1
        end
    end

    return sim
end

@inline function compute_term_sim(method::TFIDF, term_idf, q_weight, c_weight, doc_len, doc_avg)
    sim = compute_tf(method, c_weight, doc_len, doc_avg) * term_idf
    # println("==> sim=$(sim); tf=$(compute_tf(method, c_weight, doc_len, doc_avg)); idf=$(term_idf)")

    if method.add_query_freq
        sim *= compute_tf(method, q_weight, 1, doc_avg)
    end

    return sim
end


function generate_bow_vector(method::TFIDF, report::Report, df_vec::Vector{Float64}, vocab::Vocab, ngram::Int64)
    return transform_BOW(report, vocab::Vocab, ngram::Int64)
end
normalize_vec(method::TFIDF, bow::Vector{Pair{UInt32,Float64}}, df_vec::Vector{Float64})= bow
compute_final_similarity!(partial_similarities::Vector{Float64}, method::TFIDF, query::Report, candidates::Vector{Report}, df_vec::Vector{Float64}) = partial_similarities


####################################################################
# Cosine Simarity + TFIDF
####################################################################
using LinearAlgebra

mutable struct CosineTFIDF <: IRMethod
    tfidf::TFIDF
    precompute_unit_vec::Bool
end

function CosineTFIDF(tf_scheme_nm::String, idf_scheme_nm::String, precompute_unit_vec=false) 
    tf_idf = TFIDF(tf_scheme_nm, idf_scheme_nm, true, false)
    
    return CosineTFIDF(tf_idf, precompute_unit_vec)
end

computeDocFreq(method::CosineTFIDF, docFreq::DocFreq) = computeDocFreq(method.tfidf, docFreq)

function similarity(method::CosineTFIDF, q_bow, c_bow, idf_vec, cand_len)
    q_idx = 1
    c_idx = 1
    sim = 0.0

    q_len = length(q_bow) + 1
    c_len = length(c_bow) + 1
    squared_sum_q_weight = 0.0
    squared_sum_c_weight = 0.0

    @inbounds while q_idx != q_len && c_idx != c_len
        q_func_id, q_tf = q_bow[q_idx]
        c_func_id, c_tf = c_bow[c_idx]


        if q_func_id == c_func_id
            # Query and candidate contain the same function. Align positions starting from the left
            q_weight = compute_tf(method.tfidf, q_tf, 0, 0.0) * idf_vec[q_func_id]
            c_weight = compute_tf(method.tfidf, c_tf, 0, 0.0) * idf_vec[c_func_id]

            sim+= q_weight * c_weight

            squared_sum_q_weight += q_weight^2
            squared_sum_c_weight += c_weight^2

            q_idx +=1
            c_idx +=1
        elseif q_func_id > c_func_id
            # c_func_id does not exist in the query
            c_weight = compute_tf(method.tfidf, c_tf, 0, 0.0) * idf_vec[c_func_id]
            squared_sum_c_weight += c_weight^2

            c_idx += 1
        else
            # q_func_id does not exist in the candidate
            q_weight = compute_tf(method.tfidf, q_tf, 0, 0.0) * idf_vec[q_func_id]
            squared_sum_q_weight += q_weight^2

            q_idx += 1
        end
    end

    @inbounds for j=q_idx:length(q_bow)
        func_id, tf = q_bow[j]
        weight = compute_tf(method.tfidf, tf, 0, 0.0) * idf_vec[func_id]
        squared_sum_q_weight += weight^2
    end

    @inbounds for j=c_idx:length(c_bow)
        func_id, tf = c_bow[j]
        weight = compute_tf(method.tfidf, tf, 0, 0.0) * idf_vec[func_id]
        squared_sum_c_weight += weight^2
    end

    return sim/(sqrt(squared_sum_q_weight) * sqrt(squared_sum_c_weight))
end



@inline function compute_term_sim(method::CosineTFIDF, term_idf, q_weight, c_weight, doc_len, doc_avg)
    if method.precompute_unit_vec
        return q_weight * c_weight
    else
        qtf = compute_tf(method.tfidf, q_weight, doc_len, doc_avg)
        ctf = compute_tf(method.tfidf, c_weight, doc_len, doc_avg)
    
        # println("===> $(qtf * term_idf^2 * ctf) $(qtf) $(term_idf) $(ctf)")
        return qtf * term_idf^2 * ctf
    end
end


function normalize_vec(method::CosineTFIDF, bow::Vector{Pair{UInt32,Float64}}, df_vec::Vector{Float64})
    if !method.precompute_unit_vec
        return bow
    end

    # Compute unit vector
    weights = Vector{Float64}(undef, length(bow))

    # println("Report: $(report.id)")
    @inbounds for idx in eachindex(weights)
        # println("idf= $(df_vec[bow[idx].first]); tf=$(compute_tf(method.tfidf, bow[idx].second, 0, 0.0))")
        df = df_vec[bow[idx].first] > 0.0 ? df_vec[bow[idx].first] : 1.0e-100
        weights[idx] = df * compute_tf(method.tfidf, bow[idx].second, 0, 0.0)
    end

    magnitude = norm(weights)
    # println("magnitude=$(magnitude)")
    bow_norm = Vector{Pair{UInt32,Float64}}(undef, length(bow))

    for idx in eachindex(weights)
        @inbounds bow_norm[idx] = Pair{UInt32,Float64}(bow[idx].first, weights[idx] / magnitude)
    end
    # println(bow_norm)

    return bow_norm
end

function magnitude(method::CosineTFIDF, report::Report, df_vec::Vector{Float64})
    magnitude = 0.0
    bow = report.bow
    
    # println("# $(report.id)")
    @inbounds for idx in eachindex(bow)
        df = df_vec[bow[idx].first] > 0.0 ? df_vec[bow[idx].first] : 1.0e-100
        magnitude+= (df * compute_tf(method.tfidf, bow[idx].second, 0, 0.0))^2
    end

    return sqrt(magnitude)
end

@inline function compute_final_similarity!(partial_similarities::Vector{Float64}, method::CosineTFIDF, query::Report, candidates::Vector{Report}, df_vec::Vector{Float64})
    # Compute magnitude of query
    if method.precompute_unit_vec
        return partial_similarities
    end

    q_magnitude = magnitude(method, query, df_vec)
    
    Threads.@threads for idx in eachindex(partial_similarities)
        @inbounds if partial_similarities[idx] != 0.0            
            c_magnitude = magnitude(method, candidates[idx], df_vec)
            partial_similarities[idx] /=  c_magnitude * q_magnitude
        end
    end

    return partial_similarities
end


