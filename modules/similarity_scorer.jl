module similarity_scorer

export createScorer, computeSimilarity, Scorer, updateIndex!, MlScorer, PositionPosting, BasicScorer, ProposedScorer

using util
using aggregation_strategy
using Method
using DataStructures


abstract type Scorer end

struct PositionPosting
    report::Report
    positions::Vector{Int16}
end

struct Posting
    report::Report
    freq::Float64
end

@inline function get_dfcache(method, doc_freq::DocFreq)
    if doc_freq.nDocs != doc_freq.previous_n_docs
        doc_freq.cache = computeDocFreq(method, doc_freq)   
        doc_freq.previous_n_docs = doc_freq.nDocs
    end

    return doc_freq.cache
end



#################################################
# Basic
#################################################

struct BasicScorer{M <: MethodType} <: Scorer
    aggType::AggregationType
    method::M
end


function updateIndex!(scorer::BasicScorer, query::Report, doc_freq::DocFreq)
    nothing
end



function computeSimilarity(scorer::BasicScorer, query::Report, candidates::Vector{Report}, docFreq::DocFreq)
    results = fill(-9999999.99, length(candidates))
    df_vec = get_dfcache(scorer.method, docFreq)

    if length(query.stacks) == 0
        @info "Query $(query.id) is empty"
        return results
    end
    

    Threads.@threads for report_idx = 1:length(candidates)
        candidate = candidates[report_idx]
        
        if length(candidate.stacks) != 0
            agg = Aggregation(scorer.aggType, length(query.stacks), length(candidate.stacks))

            for (q_stack_idx, q_stack) in enumerate(query.stacks)
                for (c_stack_idx, c_stack) in enumerate(candidate.stacks)
                    sim = similarity(scorer.method, q_stack, c_stack, df_vec)
                    update_score_matrix!(agg, q_stack_idx, c_stack_idx, sim)
                end
            end

            results[report_idx] = aggregate(agg)
        end
    end

    return results
end

#################################################
# TraceSim Scorer - Reduce the algorithm runtime
#################################################

struct TraceSimScorer <: Scorer
    aggType::AggregationType
    method::TraceSim
end

function updateIndex!(scorer::TraceSimScorer, query::Report, doc_freq::DocFreq)
    nothing
end


"""
For each stack, create a vector that the indices of the sorted vector. 
"""
function generate_sorted_indices(report::Report)
    if length(report.stats.sorted_idxs) > 0
        return report.stats.sorted_idxs
    end 

    vec = report.stats.sorted_idxs

    for stack in report.stacks
        push!(vec, sortperm(stack))
    end

    return vec
end

function computeSimilarity(scorer::TraceSimScorer, query::Report, candidates::Vector{Report}, docFreq::DocFreq)
    results = fill(-9999999.99, length(candidates))
    df_vec = get_dfcache(scorer.method, docFreq)

    if length(query.stacks) == 0
        @info "Query $(query.id) is empty"
        return results
    end

    q_weights  = Vector{Vector{Float64}}()

    for q_stack in query.stacks
        push!(q_weights, compute_frame_weights(scorer.method, q_stack, df_vec))
    end

    q_sorted_idxs = generate_sorted_indices(query)

    Threads.@threads for report_idx = 1:length(candidates)
        candidate = candidates[report_idx]
        
        if length(candidate.stacks) != 0
            agg = Aggregation(scorer.aggType, length(query.stacks), length(candidate.stacks))
            c_sorted_idxs  = generate_sorted_indices(candidate)

            for (q_stack_idx, q_stack) in enumerate(query.stacks)
                for (c_stack_idx, c_stack) in enumerate(candidate.stacks)
                    sim = similarity(scorer.method, q_stack, c_stack, q_weights[q_stack_idx], q_sorted_idxs[q_stack_idx], c_sorted_idxs[c_stack_idx], df_vec)
                    update_score_matrix!(agg, q_stack_idx, c_stack_idx, sim)
                end
            end

            results[report_idx] = aggregate(agg)
        end
    end

    return results
end

#################################################
# Quick Align without Inverted Index 
#################################################

struct FaSTScorer <: Scorer
    method::FaST
    aggType::AggregationType
end

function get_data_structure(scorer::FaSTScorer, report::Report)
    return aggregatePositionByFuncKeepDim(report)
end

function updateIndex!(scorer::FaSTScorer, new_report::Report, doc_freq::DocFreq)
    new_report.v_positional_bow = get_data_structure(scorer, new_report)
end

function computeSimilarity(scorer::FaSTScorer, query::Report, candidates::Vector{Report}, docFreq::DocFreq)
    similarities = fill(-9999999.99, length(candidates))
    df_vec = get_dfcache(scorer.method, docFreq)

    if length(query.stacks) == 0
        @info "Query $(query.id) is empty"
        return similarities
    end

    q_bow = get_data_structure(scorer, query)
    

    Threads.@threads for report_idx = 1:length(candidates)
        candidate = candidates[report_idx]
        
        if length(candidate.stacks) != 0
            agg = Aggregation(scorer.aggType, length(q_bow), length(candidate.v_positional_bow))

            for (q_stack_idx, q_pos_bow) in enumerate(q_bow)
                for (c_stack_idx, c_pos_bow) in enumerate(candidate.v_positional_bow)
                    sim = similarity(scorer.method, q_pos_bow, c_pos_bow, df_vec)
                    update_score_matrix!(agg, q_stack_idx, c_stack_idx, sim)
                end
            end
            
            similarities[report_idx] = aggregate(agg)
        end
    end

    return similarities
end

#################################################
# Information Retrieval
#################################################

struct PlainIRScorer{M <: MethodType} <: Scorer
    method::M
    aggType::AggregationType
    ngram::Int32
end

function transform_BOW_dim(method, report::Report, ngram, vocab)::Vector{Vector{Pair{UInt32, Float64}}}
    v_bow = Vector{Vector{Pair{UInt32, Float64}}}(undef, length(report.stacks))

    for (idx, stack) in enumerate(report.stacks)
        bow = Dict{UInt32, Float64}()

        for (pos, func_id) in enumerate(stack)
            if haskey(bow, func_id)
                bow[func_id] += 1.0
            else
                bow[func_id] = 1.0
            end
        end
        
        for ngram_tkn in generate_ngrams(report.stacks, ngram)
            ngram_id = vocab.vocab[ngram_tkn]
            if haskey(bow, ngram_id)
                bow[ngram_id] += 1.0
            else
                bow[ngram_id] = 1.0
            end
        end
    
        v_bow[idx] = sort!(collect(bow))
    end

    return v_bow
end

function generate_bow_vector_dim!(scorer::PlainIRScorer, report::Report, vocab::Vocab, df_vec::Vector{Float64})
    bow = transform_BOW_dim(scorer.method, report, scorer.ngram, vocab)
    report.v_bow = bow
    report.stats.doc_lens = [length(stack) for stack in report.stacks ]



    return report.v_bow
end


function updateIndex!(scorer::PlainIRScorer, new_report::Report, doc_freq::DocFreq)
    generate_bow_vector_dim!(scorer, new_report, doc_freq.vocab, get_dfcache(scorer.method, doc_freq))
end


function computeSimilarity(scorer::PlainIRScorer, query::Report, candidates::Vector{Report}, docFreq::DocFreq)
    q_v_bow = generate_bow_vector_dim!(scorer, query, docFreq.vocab, get_dfcache(scorer.method, docFreq))    
    similarities = fill(-9999999.99, length(candidates))
    df_vec = get_dfcache(scorer.method, docFreq)

    if length(query.stacks) == 0
        @info "Query $(query.id) is empty"
        return similarities
    end
    

    Threads.@threads for report_idx = 1:length(candidates)
        candidate = candidates[report_idx]
        
        if length(candidate.stacks) != 0
            agg = Aggregation(scorer.aggType, length(q_v_bow), length(candidate.v_bow))
            for (q_stack_idx, q_bow) in enumerate(q_v_bow)
                for (c_stack_idx, c_bow) in enumerate(candidate.v_bow)
                    sim = similarity(scorer.method, q_bow, c_bow, df_vec, candidate.stats.doc_lens[c_stack_idx])
                    update_score_matrix!(agg, q_stack_idx, c_stack_idx, sim,)
                end
            end

            similarities[report_idx] = aggregate(agg)
        end
    end

    return similarities
end



#################################################
# Scorer that used our method to compute the similarity
#################################################

mutable struct ProposedScorer{M <: MethodType} <: Scorer
    method::M
    index::Dict{UInt32,Vector{PositionPosting}}
    is_sum_weights::Bool
end


function compute_weight_sum!( new_report::Report, scorer::ProposedScorer, function2positions::Vector{Pair{UInt32, Vector{Int16}}}, df_vec::Vector{Float64})
    new_report.stats.weight_sum != -Inf && return

    weight_sum = 0.0

    for (func_id, positions) in function2positions
        @inbounds df = df_vec[func_id]

        for pos in positions
            w = frame_weight(scorer.method, pos, df)
            weight_sum += w
        end
    end

    new_report.stats.weight_sum = weight_sum
end



function updateIndex!(scorer::ProposedScorer, new_report::Report, doc_freq::DocFreq)
    index = scorer.index
    df_vec = get_dfcache(scorer.method, doc_freq)

    function2positions = aggregatePositionByFunction(new_report)

    for (func_id, positions) in function2positions
        postingList = haskey(index, func_id) ? index[func_id] : Vector{PositionPosting}()

        if length(postingList) == 0
            index[func_id] = postingList
        end
        
        push!(postingList, PositionPosting(new_report, positions))
    end

    if scorer.is_sum_weights
        compute_weight_sum!(new_report, scorer, function2positions, df_vec)
    end
    
end

function computeSimilarity(scorer::ProposedScorer, query::Report, candidates::Vector{Report}, docFreq::DocFreq)
    index = scorer.index
    method = scorer.method
    q_bow = aggregatePositionByFunction(query)
    df_vec =  get_dfcache(scorer.method, docFreq)

    candidate2idx = Dict{Int64,Int64}()
    sizehint!(candidate2idx, length(candidates))
    
    for (cand_idx, cand) in  enumerate(candidates)
        candidate2idx[cand.id] = cand_idx
    end
    
    if scorer.is_sum_weights
        compute_weight_sum!(query, scorer, q_bow, df_vec)
    end
    
    stub = Vector{Posting}()
    similarities = zeros(length(candidates))
   
    for token_idx = 1:length(q_bow)
        func_id, q_positions = q_bow[token_idx]

        postingList = get(index, func_id, stub)
        ndf = df_vec[func_id]

        length(postingList) == 0 && continue

        Threads.@threads for posting in postingList
            candidate_idx = get(candidate2idx, posting.report.id, -1)
            candidate_idx < 0 && continue

            c_positions = posting.positions            
            score = 0.0
            j = 1
            min_len_positions = length(q_positions) < length(c_positions) ? length(q_positions) + 1 : length(c_positions) + 1

            @inbounds while j != min_len_positions
                c_gap_score = gap_score(method, c_positions[j], ndf)
                q_gap_score = gap_score(method, q_positions[j], ndf) 

                score -= (q_gap_score + c_gap_score)
                m = match_score(method, q_positions[j], c_positions[j], ndf, -q_gap_score, -c_gap_score)
                score += m
                # println("#Cand:$(candidate_idx)\tfunc: $(func_id)\tc_pos: $(c_positions[j]), q_pos: $(q_positions[j]), ndf: $(ndf)")
                # println("===> c_gap_score: $(c_gap_score), q_gap_score: $(q_gap_score), m: $(m)")

                j += 1
            end

            similarities[candidate_idx] += score
        end
    end

    return compute_final_similarity!(similarities, method, query, candidates, df_vec)
end


#########################################################
# Information retrieval techniques
########################################################
mutable struct IRScorer{M <: IRMethod} <: Scorer
    method::M
    index::Dict{UInt32,Vector{Posting}}
    doc_sum::Float64
    n_docs::Float64
    ngram::Int32
end

function get_posting_list!(index, func_id)
    postingList = haskey(index, func_id) ? index[func_id] : Vector{Posting}()

    if length(postingList) == 0
        index[func_id] = postingList
    end

    return postingList
end

function transform_BOW(method, report::Report, ngram, vocab)::Vector{Pair{UInt32, Float64}}
    bow = Dict{UInt32, Float64}()

    for stack in report.stacks
        for (pos, func_id) in enumerate(stack)
            if haskey(bow, func_id)
                bow[func_id] += 1.0
            else
                bow[func_id] = 1.0
            end
        end
    end

    for ngram_tkn in generate_ngrams(report.stacks, ngram)
        ngram_id = vocab.vocab[ngram_tkn]
        if haskey(bow, ngram_id)
            bow[ngram_id] += 1.0
        else
            bow[ngram_id] = 1.0
        end
    end

    return collect(bow)
end

function generate_bow_vector!(scorer::IRScorer, report::Report, vocab::Vocab, df_vec::Vector{Float64})
    bow = transform_BOW(scorer.method, report, scorer.ngram, vocab)
    report.bow = normalize_vec(scorer.method, bow, df_vec)

    return report.bow
end


function updateIndex!(scorer::IRScorer, new_report::Report, doc_freq::DocFreq)
    index = scorer.index
    weighted_bow = generate_bow_vector!(scorer, new_report, doc_freq.vocab, get_dfcache(scorer.method, doc_freq))
    new_report.bow = weighted_bow

    for idx in eachindex(weighted_bow)
        func_id, term_weight = weighted_bow[idx]
        postingList = get_posting_list!(index, func_id)

        push!(postingList, Posting(new_report, term_weight))
    end

    doc_len = set_doc_len!(new_report)
    scorer.doc_sum += doc_len
    scorer.n_docs += 1
end


function computeSimilarity(scorer::IRScorer, query::Report, candidates::Vector{Report}, docFreq::DocFreq)
    index = scorer.index
    method = scorer.method
    q_bow = generate_bow_vector!(scorer, query, docFreq.vocab, get_dfcache(scorer.method, docFreq))
    
    candidate2idx = Dict{Int64,Int64}(map(k -> (k[2].id, k[1]), enumerate(candidates)))
    idf_vec =  get_dfcache(scorer.method, docFreq)
    doc_avg = scorer.doc_sum/scorer.n_docs
    
    # number_of_match = zeros(length(candidates))
    # lengths = [ cand.stats.doc_len for cand in candidates]
    
    stub = Vector{Posting}()
    similarities = zeros(length(candidates))
   
    for token_idx = 1:length(q_bow)
    # for token_idx = 1:length(q_bow)
        func_id, q_freq = q_bow[token_idx]
        
        posting_list = get(index, func_id, stub)
        idf = idf_vec[func_id]

        length(posting_list) == 0 && continue

        Threads.@threads for posting in posting_list            
            candidate_idx = get(candidate2idx, posting.report.id, -1)
            candidate_idx < 0 && continue
            
            # println("#Cand:$(posting.report.id)\tfunc: $(func_id)\t q_freq=$(q_freq)\tc_freq=$(posting.freq)")
            # println("\t$(compute_term_sim(method, idf, q_freq, posting.freq, posting.report.stats.doc_len , doc_avg))")
            similarities[candidate_idx] += compute_term_sim(method, idf, q_freq, posting.freq, posting.report.stats.doc_len , doc_avg)
            # number_of_match[candidate_idx]+=1.0                
        end
    end

    # mean_match = sum([n/l for (n,l) in zip(number_of_match,lengths)])/ length(candidates)
    # println("Mean $(mean_match)")

    return compute_final_similarity!(similarities, method, query, candidates, idf_vec)
end


#########################################################
# Create Scorer function
########################################################


@inline function createScorer(methodName::String, args, upt_doc_freq)::Scorer
    if methodName == "trace_sim"
        @info("TraceSim")
        aggStrategy = parseAggStrategy(args[:aggregate])
        method = TraceSim(args[:alpha], args[:beta], args[:gamma],
            get(args, :sigmoid, false), 
            get(args, :b, 0.0),
            get(args, :idf, false),
            get(args, :reciprocal_func, true),
            get(args, :no_norm, false),
            get(args, :sum_match, false),
            )

        return TraceSimScorer(aggStrategy, method)
    elseif methodName == "pdm"
        @info("PDM")

        aggStrategy = parseAggStrategy(args[:aggregate])
        method = PDM(args[:c], args[:o])

        return BasicScorer{PDM}(aggStrategy, method)
    elseif methodName == "prefix_match"
        @info("PrefixMatch")
        aggStrategy = parseAggStrategy(args[:aggregate])
        method = PrefixMatch()

        return BasicScorer{PrefixMatch}(aggStrategy, method)
    elseif methodName == "fast"
        @info("FaST")
        
        method = FaST(args[:alpha], args[:beta], args[:gamma], upt_doc_freq)
        return ProposedScorer{FaST}(method, Dict{UInt32,MutableLinkedList{PositionPosting}}(), !upt_doc_freq) 
    elseif methodName == "plain_fast"
        @info("FaST Without Inverted Index")
        aggStrategy = parseAggStrategy(args[:aggregate])
        method = FaST(args[:alpha], args[:beta], args[:gamma], upt_doc_freq)

        return FaSTScorer(method, aggStrategy) 
    elseif methodName == "plain_tf_idf"
        @info("TF-IDF Without Inverted Index")
        aggStrategy = parseAggStrategy(args[:aggregate])
        
        method = TFIDF(get(args, :tf_scheme, "sqrt"), get(args, :idf_scheme, "smooth"),  false, get(args, :add_query_freq, false))

        return PlainIRScorer{TFIDF}(method, aggStrategy, get(args, :ngram, 1)) 
    elseif methodName == "plain_cosine_tf_idf"
        @info("Cosine Without Inverted Index")
        aggStrategy = parseAggStrategy(args[:aggregate])
        method = CosineTFIDF(get(args, :tf_scheme, "sqrt"), get(args, :idf_scheme, "smooth"),  !upt_doc_freq)

        return PlainIRScorer{CosineTFIDF}(method, aggStrategy, get(args, :ngram, 1)) 
    elseif methodName == "tf_idf"
        method = TFIDF(get(args, :tf_scheme, "sqrt"), get(args, :idf_scheme, "smooth"),  false, get(args, :add_query_freq, false))
        return IRScorer{TFIDF}(method, Dict{UInt32,Vector{Posting}}(), 0, 0, get(args, :ngram, 1)) 
    elseif methodName == "bm25"
        method = BM25(arg[:k1], args[:b], 0.0)
        return IRScorer{BM25}(method, Dict{UInt32,Vector{Posting}}(), 0, 0, get(args, :ngram, 1)) 
    elseif methodName == "cosine_tf_idf"
        @info("CosineTFIDF")
        method = CosineTFIDF(get(args, :tf_scheme, "sqrt"), get(args, :idf_scheme, "smooth"), !upt_doc_freq)
        return IRScorer{CosineTFIDF}(method, Dict{UInt32,Vector{Posting}}(), 0, 0, get(args, :ngram, 1)) 
    elseif methodName == "fast_pdm"
        @info("Fast PDM")
        method = PDM(args[:c], args[:o])
        return ProposedScorer{PDM}(method, Dict{UInt32,MutableLinkedList{PositionPosting}}(), false)
    elseif methodName == "fast_trace_sim"
        @info("Fast TraceSim")
        method = TraceSim(args[:alpha], args[:beta], args[:gamma],
            get(args, :sigmoid, false), 
            get(args, :b, 0.0),
            get(args, :idf, false),
            get(args, :reciprocal_func, true),
            get(args, :no_norm, false)
            )

        return ProposedScorer{TraceSim}(method, Dict{UInt32,MutableLinkedList{PositionPosting}}(), false)
    end
end


end