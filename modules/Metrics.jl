module Metrics

using util
using DataStructures
using PyCall

export createMAP_RecallRate, createBinaryPredictionROC, reset, update_metric!, compute, MAP_RecallRate, BinaryPredictionROC

abstract type Metric end

mutable struct MAP_RecallRate <: Metric
    label::String
    k::Int
    sum_map::Float64
    hits_per_k::Dict{Int, Float64}
    n_duplicates::Int
    group_by_master::Bool
end

function createMAP_RecallRate(;label="MAP_RecallRate", k=20, group_by_master=true)
    return MAP_RecallRate(label, k, 0.0, Dict{Int, Float64}([ (i, 0.0) for i in 1:k]), 0, group_by_master)
end

function reset(metric::MAP_RecallRate)
    metric.hits_per_k = Dict{Int, Float64}([ (i, 0.0) for i in 1:k])
    metric.n_duplicates = 0
    metric.sum_map = 0.0
end

function update_metric!(metric::MAP_RecallRate, query::Report, recommendation_list::Vector{Tuple{Report, Float64}})
    if query.dup_id == query.id
        #println("Simgleton: $(query.id) $([ (r.id, s) for (r, s)  in recommendation_list[1:20]])")
        return
    end 

    buckets = Set()
    pos = Inf
    correct_cand = Int64(-1)

    for (cand, score) in recommendation_list
        if metric.group_by_master
            query.dup_id in buckets && continue
            push!(buckets, cand.dup_id)
        else
            push!(buckets, cand.id)
        end
            
        if cand.dup_id == query.dup_id
            pos = Float64(length(buckets))

            
            # println("$(query.id) $(cand.id) $(pos) $([(r.id, s) for (r, s)  in recommendation_list[1:20]])")
            # @debug 
            # print("{}, {}, {}, {}, {}".format(report_id, pos, cand_id, self.master_id_by_bug_id[report_id], list(
            #     zip(candidates[:30], map(lambda x: self.master_id_by_bug_id[x], candidates[:30]), scores[:30]))))
            correct_cand = cand.id
            break
        end
    end

    for k in 1:metric.k
        k < pos && continue

        metric.hits_per_k[k]+=1.0
    end

    metric.sum_map += 1.0/pos
    metric.n_duplicates += 1

    return pos    
end


function compute(metric::MAP_RecallRate)        
    recall_rate = Dict{Int,Float64}()

    for k in 1:metric.k
        recall_rate[k] = metric.hits_per_k[k] / metric.n_duplicates
    end

    return Dict("type" => "metric",
            "label" => metric.label,
            "hits_per_k" => metric.hits_per_k,
            "rr" => recall_rate,
            "sum_map" => metric.sum_map,
            "map"=> metric.sum_map / metric.n_duplicates,
            "total" => metric.n_duplicates,
            )
end


mutable struct BinaryPredictionROC <: Metric
    label::String
    queries::MutableLinkedList{Report}
    scores::MutableLinkedList{Float64}
end

function  createBinaryPredictionROC(;label="BinaryPredictionROC")
    return BinaryPredictionROC(label, MutableLinkedList{Report}(), MutableLinkedList{Float64}())
end

function reset!(metric::BinaryPredictionROC)
    metric.queries = MutableLinkedList{Report}()
    metric.scores = MutableLinkedList{Float64}()
end


function update_metric!(metric::BinaryPredictionROC, query::Report, recommendation_list::Vector{Tuple{Report, Float64}})
    top_candidate = recommendation_list[1][1]
    top_score = recommendation_list[1][2]

    push!(metric.queries, query)
    push!(metric.scores, top_score)

        # print("{} {} {} {}".format(query_id, top_candidate, top_score, self.master_id_by_bug_id[query_id] != query_id))

    # print("{}, {}, {}, {}, {}, {}, {}".format(
    #     self.master_id_by_bug_id[query_id] == query_id, query_id,
    #     self.master_id_by_bug_id[query_id], top_candidate, self.master_id_by_bug_id[top_candidate], top_score,
    #     [(k, s) for k, s in zip(candidates[:20], scores[:20])]))

end

function compute(metric::BinaryPredictionROC)
    SKLEARN_METRICS = pyimport("sklearn.metrics")
    y_true = [Int(query.dup_id != query.id) for query in metric.queries]
    fpr, tpr, thresholds = SKLEARN_METRICS.roc_curve(y_true, collect(metric.scores), pos_label=1)

    return Dict("type" => "metric",
                "label" => metric.label,
                "fpr" => fpr,
                "tpr" => tpr,
                "auc" => SKLEARN_METRICS.auc(fpr,tpr),
                "n_queries" =>  length(metric.queries),
                "threshold" => thresholds,
    )

end

end