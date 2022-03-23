module aggregation_strategy

export AggregationType, parseAggStrategy, update_score_matrix!, aggregate, Aggregation

using util
using Statistics

@enum AggregationType MAX AVG_QUERY AVG_CAND AVG_SHORT AVG_LONG AVG_QUERY_CAND


function parseAggStrategy(methodName)
    if methodName == "max"
        return MAX::AggregationType
    elseif methodName == "avg_query"
        return AVG_QUERY::AggregationType
    elseif methodName == "avg_cand"
        return AVG_CAND::AggregationType
    elseif methodName == "avg_short"
        return AVG_SHORT::AggregationType
    elseif methodName == "avg_long"
        return AVG_LONG::AggregationType
    elseif methodName == "avg_query_cand"
        return AVG_QUERY_CAND::AggregationType
    end
end

struct Aggregation
    aggType::AggregationType
    query_max_vec::Vector{Float64}
    cand_max_vec::Vector{Float64}
end

Aggregation(aggType::AggregationType, q_len, c_len) = Aggregation(aggType, fill(typemin(Float64), q_len),  fill(typemin(Float64), c_len))

function update_score_matrix!(agg::Aggregation, query_idx, cand_idx, score)
    if agg.query_max_vec[query_idx] < score
        agg.query_max_vec[query_idx] = score
    end

    if agg.cand_max_vec[cand_idx] < score
        agg.cand_max_vec[cand_idx] = score
    end
end

function aggregate(agg::Aggregation)::Float64
    if agg.aggType == MAX::AggregationType
        largest = typemin(Float64)

        for v in agg.query_max_vec
            if v > largest
                largest = v
            end
        end

        for v in agg.cand_max_vec
            if v > largest
                largest = v
            end
        end

        return largest
    elseif agg.aggType == AVG_QUERY::AggregationType
        return mean(agg.query_max_vec)
    elseif agg.aggType == AVG_CAND::AggregationType
        return mean(agg.cand_max_vec)
    elseif agg.aggType == AVG_SHORT::AggregationType
        querySize = length(agg.query_max_vec) 
        candSize = length(agg.cand_max_vec) 

        return querySize < candSize  ? mean(agg.query_max_vec) : mean(agg.cand_max_vec)
    elseif agg.aggType == AVG_LONG::AggregationType
        querySize = length(agg.query_max_vec) 
        candSize = length(agg.cand_max_vec) 

        return querySize > candSize  ? mean(agg.query_max_vec) : mean(agg.cand_max_vec)
    elseif agg.aggType == AVG_QUERY_CAND::AggregationType
        return (mean(agg.query_max_vec) + mean(agg.cand_max_vec)) / 2.0
    end
    
    return typemax(Float64)
end


end