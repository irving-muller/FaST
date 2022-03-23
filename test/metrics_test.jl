using Test
using Metrics
using util

function createRecommendationList(correct_pos; range=100)
    extra_id = range * 2
    dup_id = extra_id + 2
    query = Report(extra_id + 1, dup_id, 1000, 10, [[0]])

    recommendation_list= Vector{Tuple{Report, Float64}}(undef, range)
    recommendation_list[1] =  (Report(1, 1, 1000, 10, [[0]]), 100.0)
    recommendation_list[2] =  (Report(extra_id+3, 1, 1000, 10, [[0]]), 100.0)

    recommendation_list[3] =  (Report(2, 2, 1000, 10, [[0]]), 100.0)
    recommendation_list[4] =  (Report(extra_id+4, 2, 1000, 10, [[0]]), 100.0)
    recommendation_list[5] =  (Report(extra_id+5, 2, 1000, 10, [[0]]), 100.0)

    recommendation_list[6] =  (Report(3, 3, 1000, 10, [[0]]), 100.0)

    recommendation_list[7] =  (Report(4, 1, 1000, 10, [[0]]), 100.0)
    recommendation_list[8] =  (Report(extra_id+6, 1, 1000, 10, [[0]]), 100.0)
    recommendation_list[9] =  (Report(extra_id+7, 1, 1000, 10, [[0]]), 100.0)

    for i in 9:range
        recommendation_list[i] = (Report(i, i, 1000, 10, [[0]]), 100.0)
    end

    if correct_pos > 0
        recommendation_list[correct_pos] = (Report(extra_id+8, dup_id, 1000, 10, [[0]]), 100.0)
    end
    
    return (query, recommendation_list)
end

map_rr = createMAP_RecallRate(;group_by_master=false)

correct_pos_lis = [1,1,1,2,2,2,3,3,3,4,4,4,5,5,6,7,19,20,20,21,50,60,44,43,80,99, -1]




for correct_pos in correct_pos_lis
    # println("Correct: $(correct_pos)")
    
    query1, recommendation_list = createRecommendationList(correct_pos)
    # println("\tdup_id: $(recommendation_list[correct_pos][1].dup_id)")
    # println("\tdup_id: $([recommendation_list[i][1].dup_id for i in 1:length(recommendation_list)])")
    update_metric!(map_rr, query1, recommendation_list)
end

result = compute(map_rr)

rr_values = Vector{Float64}(undef, 20)

for (k,v) in result["rr"]
    rr_values[k] = v
end

@test rr_values == [3.0,6.0,9.0,12.0,14.0,15.0,16.0,16.0,16.0,16.0,16.0,16.0,16.0,16.0,16.0,16.0,16.0,16.0,17.0, 19.0]./length(correct_pos_lis)
@test result["map"] == 0.2690750073903208

map_rr = createMAP_RecallRate(;group_by_master=true)

for correct_pos in correct_pos_lis    
    query, recommendation_list = createRecommendationList(correct_pos)
    update_metric!(map_rr, query, recommendation_list)
end
result = compute(map_rr)

rr_values = Vector{Float64}(undef, 20)

for (k,v) in result["rr"]
    rr_values[k] = v
end

@test rr_values == [3.0,9.0,15.0,16.0,16.0,16.0,16.0,16.0,16.0,16.0,16.0,16.0,16.0,17.0,19.0,20.0,20.0,20.0,20.0, 20.0]./length(correct_pos_lis)
@test result["map"] == 0.3197627552402343

