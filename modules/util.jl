module util

export Vocab, setdefault!, Report, submittedEarlier, calculateDayTimestamp, DocFreq, updateDocFreq!,isduplicate, get_basic_args, aggregatePositionByFunction, sigmoid, set_doc_len!, generate_ngrams, add_ngram_to_vocab!, vocabLength, aggregatePositionByFuncKeepDim

using ArgParse
using Dates
using JSON
using DataStructures


sigmoid(x)=1.0/(1.0+exp(-x))


struct Vocab
    vocab::Dict{String, UInt32}
    ukn_set::Set{UInt32}
end


function vocabLength(vocab::Vocab)::Int64
    return length(vocab.vocab)
end


function setdefault!(vocab::Vocab, functionName::String, isUkn=false)
    tkn_id = get!(vocab.vocab, functionName, vocabLength(vocab) + 1)

    if isUkn
        push!(vocab.ukn_set, tkn_id)
    end

    return tkn_id
end

mutable struct  ReportStats
    doc_len::Int64
    doc_lens::Vector{Int64}
    weight_sum::Float64
    sorted_idxs::Vector{Vector{UInt16}}
end


mutable struct Report
    id::Int64
    dup_id::Int64
    creation_date::Int64
    creation_day_ts::UInt64
    stacks::Vector{Vector{UInt32}}
    bow::Vector{Pair{UInt32,Float64}}
    positional_bow::Vector{Pair{UInt32, Vector{Int16}}}


    v_bow::Vector{Vector{Pair{UInt32,Float64}}}
    v_positional_bow::Vector{Vector{Pair{UInt32, Vector{Int16}}}}

    stats::ReportStats
end

@inline function Report(id, dup_id, creation_date, creation_day_ts, stacks)
    return Report(id, dup_id, creation_date, creation_day_ts, stacks, Vector{Pair{UInt32,Float64}}(), Vector{Pair{UInt32, Vector{Int16}}}(), Vector{Vector{Pair{UInt32,Float64}}}(), Vector{Vector{Pair{UInt32, Vector{Int16}}}}(), ReportStats(-999999999, Vector{Int64}(), -Inf, Vector{Vector{UInt16}}()))
end

isduplicate(report::Report) = report.id != report.dup_id

submittedEarlier(r1::Report, r2::Report) = r1.creation_date < r2.creation_date || (r1.creation_date == r2.creation_date && r1.id < r2.id)

function calculateDayTimestamp(date)
    return UInt64(floor(date / (24 * 60 * 60)))
end

function set_doc_len!(report::Report)
    doc_len = 0

    for stack in report.stacks
        doc_len += length(stack)
    end

    report.stats.doc_len = doc_len

    return doc_len
end


mutable struct DocFreq
    doc_freq::Vector{Float64} 
    nDocs::Float64
    freq_by_stacks::Bool
    static_df_ukn::Bool
    vocab::Vocab
    
    cache::Vector{Float64}
    previous_n_docs::Float64
end

DocFreq(doc_freq::Vector{Float64}, nDocs::Float64, freq_by_stacks::Bool, static_df_ukn::Bool, vocab::Vocab) = DocFreq(doc_freq, nDocs, freq_by_stacks, static_df_ukn, vocab, Vector{Float64}(), 0)


function updateDocFreq!(docfreqObj::DocFreq, reports)
    for report in reports
        if docfreqObj.freq_by_stacks
            for stack in report.stacks
                for funcId in Set(stack)
                    docfreqObj.doc_freq[funcId] += 1
                end
                docfreqObj.nDocs += 1
            end
        else
            function_set = Set(( funcId for stack in report.stacks for funcId in stack))
            docfreqObj.nDocs += 1

            for funcId in function_set
                docfreqObj.doc_freq[funcId] += 1
            end
        end
    end

    if docfreqObj.static_df_ukn
        for funcId in docfreqObj.vocab.ukn_set
            docfreqObj.doc_freq[funcId]= docfreqObj.nDocs - 1
        end
    end
end

function aggregatePositionByFunction(report::Report)::Vector{Pair{UInt32, Vector{Int16}}}
    bow = Dict{UInt32, Vector{Int16}}()

    for stack in report.stacks
        for (pos, func_id) in enumerate(stack)
            if haskey(bow, func_id)
                push!(bow[func_id], pos)
            else
                bow[func_id] = [pos]
            end
        end
    end

    for positions in values(bow)
        sort!(positions)
    end

    return collect(bow)
end


function aggregatePositionByFuncKeepDim(report::Report)::Vector{Vector{Pair{UInt32, Vector{Int16}}}}
    v_bow = Vector{Vector{Pair{UInt32, Vector{Int16}}}}(undef, length(report.stacks))
    
    for (idx, stack) in enumerate(report.stacks)
        bow = Dict{UInt32, Vector{Int16}}()
        for (pos, func_id) in enumerate(stack)
            if haskey(bow, func_id)
                push!(bow[func_id], pos)
            else
                bow[func_id] = [pos]
            end
        end

        for positions in values(bow)
            sort!(positions)
        end
        
        v_bow[idx] = sort!(collect(bow))
    end

    return v_bow
end


function add_ngram_to_vocab!(vocab::Vocab, reportid2report::Dict{Int64, Report}, ngram)
    for report in values(reportid2report)
        for ngram_tkn in generate_ngrams(report.stacks, ngram)
            setdefault!(vocab, ngram_tkn)
        end 
    end
end

function generate_ngrams(stacks::Vector{Vector{UInt32}}, ngram)
    ngram_vec = Vector{String}()

    for stack in stacks
        for n=2:ngram
            v = Vector{UInt32}(undef, n)

            for start_idx=1:(length(stack) - n + 1)
                p = 1
                for idx=start_idx:(start_idx + n - 1)
                    v[p] = stack[idx]
                    p+=1
                end

                push!(ngram_vec, join(v, ","))
            end

            
        end 
    end

    return ngram_vec
end

function generate_ngrams(stack::Vector{UInt32}, ngram)
    ngram_vec = Vector{String}()
    
    for n=2:ngram
        v = Vector{UInt32}(undef, n)

        for start_idx=1:(length(stack) - n + 1)
            p = 1
            for idx=start_idx:(start_idx + n - 1)
                v[p] = stack[idx]
                p+=1
            end

            push!(ngram_vec, join(v, ","))
        end

        
    end 

    return ngram_vec
end


function get_basic_args()
    s = ArgParseSettings()
    
    @add_arg_table s begin
        "--window", "-w"
            help = "Time window"
            arg_type = Int
            default = -1
        "--max_depth"
            help = ""
            arg_type = Int
            default = 300    
        "--filter_recursion"
            help = "Options: none, modani and brodie"
            default = "none" 
        "--keep_ukn"
            action = :store_true
        "--aggregate"
            help = "Options: max, avg_query, avg_cand, avg_short, avg_long avg_query_cand"
            default = "max" 
        "--static_df_ukn"
            action = :store_true    
        "--filter_func"
            help = "Options: none, top_k, top_k_trim"
            default = "none"
        "--filter_func_k"
            help = "Percentage of the k-top functions (values between 0 and 100)"
            arg_type = Float32
            default = Float32(-1.0)
        "--freq_by_stacks"
            action = :store_true  
        "--upd_doc_freq"
            action = :store_true 
        "--smoothing"
            action = :store_true 
        "--interesting_stacks"
            action = :store_true 
            help = "Following https://bazaar.launchpad.net/~bgo-maintainers/bugzilla-traceparser/3.4/view/head:/lib/TraceParser/Trace.pm, 
                we select the interesting stacks that might contain the problem. This option only works for Gnome."
    end

    s.add_help = false

    return s
end


end