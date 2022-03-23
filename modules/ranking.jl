module ranking

export initSunStrategy, SunStrategy, Bucket, evaluate

using util
using Dates
using DataStructures
using preprocessing
using similarity_scorer
using Metrics
using Method



mutable struct Bucket
    master::Report
    reports::MutableLinkedList{Report}
    newestReport::Report
end


struct SunStrategy
    buckets::Dict{Int64, Bucket}
    window::Int32
    docFreq::DocFreq
    funcRemoval::DocFreqRemoval
    updDocFreq::Bool
end


function addReport!(strategy::SunStrategy, report; updateDocFreq=true)
    #update DocFreq
    if updateDocFreq
        updateDocFreq!(strategy.docFreq, [report])
    end
    
    buckets = strategy.buckets

    if !haskey(buckets, report.dup_id)
        # New bucket
        buckets[report.dup_id] =  Bucket(report, MutableLinkedList{Report}(report), report)
    else
        bucket = buckets[report.dup_id]
        push!(bucket.reports, report)

        if submittedEarlier(bucket.newestReport, report)
            bucket.newestReport = report
        end

        if submittedEarlier(report, bucket.master)
            bucket.master = report
        end
    end
end


function initSunStrategy(queries, reportid2report::Dict{Int64, Report}, scorer, funcRemoval::DocFreqRemoval, vocab, window, freq_by_stacks, static_df_ukn, upd_doc_freq=false, smooth=false)
    oldest = reportid2report[queries[1]]

    for (idx, report_id) in enumerate(queries)
        idx == 1 && continue
        cur =  reportid2report[report_id]

        if submittedEarlier(cur, oldest)
            oldest = cur
        end
    end
    @debug "Oldest query $(oldest)"

    @info "Update document frequency: $(upd_doc_freq)"
    
    if smooth    
        @info "Smoothing Document frequency "
        docfreqObj = DocFreq(ones(length(vocab.vocab)), 1.0, freq_by_stacks, static_df_ukn, vocab)
    else
        docfreqObj = DocFreq(zeros(length(vocab.vocab)), 0.0, freq_by_stacks, static_df_ukn, vocab)
    end

    strategy = SunStrategy(Dict{Int64, Bucket}(), window, docfreqObj, funcRemoval, upd_doc_freq)
    initialDbReports = MutableLinkedList{Report}()
    bucketId2Size = Dict{Int64, Int64}()

    for report in values(reportid2report)
        !submittedEarlier(report, oldest) && continue
        
        get!(bucketId2Size, report.dup_id, 0)
        bucketId2Size[report.dup_id]+=1

        addReport!(strategy, report, updateDocFreq=false)

        push!(initialDbReports, report)
    end

    # Check buckets
    for (bucket_id, bucket) in strategy.buckets
        if bucket_id != bucket.master.id
            @error "Bucket id and master id is not the same: $(bucket_id) $(bucket.master.id)"
        end

        length(bucket.reports) != bucketId2Size[bucket_id] && @error "Bucket size is wrong: $(bucket_id) $(length(bucket.reports)) $(bucketId2Size[bucket_id])"
    end

    # Update function document frequency
    updateDocFreq!(docfreqObj, initialDbReports)

    for report in initialDbReports
        updateIndex!(scorer, report, docfreqObj)
    end

    # Remove common function in the stacktrace
    for report in initialDbReports
        removeCommonFunction!(report, strategy.funcRemoval, docfreqObj)
    end

    return strategy
end

function getCandidates(strategy::SunStrategy, query::Report)
    candidates = MutableLinkedList{Report}()

    for bucket in values(strategy.buckets)
        if strategy.window > 0 &&  strategy.window < (query.creation_day_ts - bucket.newestReport.creation_day_ts )
            continue
        end

        !submittedEarlier(bucket.newestReport, query) && continue
        
        for report in  bucket.reports
            submittedEarlier(query, report) && error("$(report.id) is in the bucket but it was submitted after query $(query.id) ")
            query.id == report.id && error("query $(query.id) is in the bucket")
            push!(candidates, report)
        end
    end

    return collect(candidates)
end 


function evaluate(strategy, queries, reportid2report::Dict{Int64, Report},  scorer, metrics)
    total = 0.0
    total_time_sim = 0.0
    total_time_update = 0.0
    
    for (idx, query_id) in enumerate(queries)
        timeScore = @elapsed begin
            query = reportid2report[query_id]
            candidates = getCandidates(strategy, query)

            removeCommonFunction!(query, strategy.funcRemoval, strategy.docFreq)
            
            for candidate in candidates
                candidate.id == query_id && error("Query $(query_id) is within candidate list")
            end

            if length(candidates) == 0
                @warn "Query $(query_id) has 0 candidates!"
            end
                 
            timesim = @elapsed scores = computeSimilarity(scorer, query, candidates, strategy.docFreq)
            recommendation_list = map((i,j)->(i,j), candidates, scores)

            # Sort  in descending order the bugs by probability of being duplicate
            sort!(recommendation_list, by=x->(x[2],x[1].id), rev=true)

            # println("$(query_id)\t$(length(recommendation_list))\t$([(recommendation_list[i][1].id, recommendation_list[i][2]) for i in 1:10])")

            for metric in metrics
                update_metric!(metric, query, recommendation_list)                
            end
            
            # Add report to buckets and add to index
            addReport!(strategy, query; updateDocFreq=strategy.updDocFreq)
            timeupdate = @elapsed updateIndex!(scorer, query, strategy.docFreq)
            
            total_time_sim += timesim
            total_time_update += timeupdate
        end

        if idx > 0 && idx % 500 == 0
            @info "It generated recommendation of $(idx) reports"
        end

        total += timeScore
    end

    @info "Time to generate recommendation of $(total) reports: $(length(queries))"
    @info "Time to related to the method: similarity=$(total_time_sim) update=$(total_time_update)   reports: $(length(queries))"
    
    return [compute(m) for m in metrics]
end

end