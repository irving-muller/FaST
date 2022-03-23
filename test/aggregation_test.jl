using Test
using DataStructures


using preprocessing
using JSON
using util
using Method
using aggregation_strategy


function aggregateMatrix(stgTypeStr, matrix)
    stgType = parseAggStrategy(stgTypeStr)
    nrow, ncol = size(matrix)
    agg = Aggregation(stgType, nrow, ncol)

    for cand_idx in 1:ncol
        for query_idx in 1:nrow
            update_score_matrix!(agg, query_idx, cand_idx, matrix[query_idx, cand_idx])
        end
    end

    return aggregate(agg)
end 



matrix = [
   0.6 0.5 0.9 -1.0 -1.0;
   0.0 -2.0 1.0 -1.0 -1.0;
   0.0 -1.0 0.0 -1.0 -1.0;
]


@test aggregateMatrix("max", matrix)==1.0
@test aggregateMatrix("avg_query", matrix)== 0.6333333333333333
@test abs(aggregateMatrix("avg_cand", matrix) - 0.02000000000) < 0.000000001
@test abs(aggregateMatrix("avg_short", matrix) - 0.63333333333333) < 0.000000001
@test abs(aggregateMatrix("avg_long", matrix) - 0.02000000000)  < 0.000000001
@test abs(aggregateMatrix("avg_query_cand", matrix) -  0.3266666666)  < 0.000000001

@test abs(aggregateMatrix("avg_query", transpose(matrix)) - 0.02000000000) < 0.000000001
@test abs(aggregateMatrix("avg_cand", transpose(matrix)) - 0.63333333333333)  < 0.000000001
@test abs(aggregateMatrix("avg_short", transpose(matrix)) - 0.63333333333333) < 0.000000001
@test abs(aggregateMatrix("avg_long", transpose(matrix)) - 0.02000000000)  < 0.000000001
@test abs(aggregateMatrix("avg_query_cand", transpose(matrix)) -   0.3266666666)  < 0.000000001




idfs = [(0x00010f8f, 0.0007444291882413274), (0x00010f90, 0.0007444291882413274), (0x00020d3a, 0.0006782577048420984), (0x00020d3a, 0.0006782577048420984), (0x00000446, 0.12638753329252758), (0x00010f91, 0.001505401247332462), (0x00010f8f, 0.0007444291882413274), (0x00010f90, 0.0007444291882413274), (0x00020d3a, 0.0006782577048420984), (0x00020d3a, 0.0006782577048420984), (0x00000446, 0.12638753329252758), (0x00010f91, 0.001505401247332462), (0x0000c9a1, 0.0010918294760872802), (0x00012afe, 0.0009264007675892074), (0x0002b716, 0.0001323429667984582), (0x0002b717, 0.0001323429667984582), (0x0002b718, 0.0001323429667984582)]
df = zeros(500000)

for (idx, df2) in idfs
    df[idx] = df2 * 100
end

query = [[0x00010f8f, 0x00010f90, 0x00020d3a, 0x00020d3a, 0x00000446, 0x00010f91, 0x00022c31, 0x00022c32, 0x0002e06c, 0x0002e06d, 0x0002e06e, 0x0002e06f, 0x0002277c, 0x000009c1, 0x00000780, 0x0000a336, 0x0000703f, 0x00006da4, 0x00011e66, 0x00006da4, 0x00006da5, 0x0000b3a0, 0x00006da6, 0x0002675a], [0x00010f8f, 0x00010f90, 0x00020d3a, 0x00020d3a, 0x00000446, 0x00010f91, 0x0000c9a1, 0x00012afe, 0x0002b716, 0x0002b717, 0x0002b718], [0x00010f8f, 0x00010f90, 0x00020d3a, 0x00020d3a, 0x00000446, 0x00010f91, 0x00005f39, 0x00007072, 0x00014fe2, 0x00014fe3, 0x00014fe4, 0x00001818, 0x000003eb, 0x0000b296, 0x00007d53, 0x0000adc2, 0x00000a60, 0x000088ad, 0x000088ae, 0x0002436b, 0x0001012b, 0x0000794a, 0x0001e4af, 0x0000bb81]]

cand = [[0x00010f8f, 0x00010f90, 0x00020d3a, 0x00020d3a, 0x00000446, 0x00010f91], 
[0x00010f8f, 0x00010f90, 0x00020d3a, 0x00020d3a, 0x00000446, 0x00010f91, 0x0000c9a1, 0x00012afe, 0x0002b716, 0x0002b717, 0x0002b718]]


tracesim = TraceSim(1.5 ,0.8, 1.0, false, 0,false, false, false)

matrix = zeros((length(query), length(cand)))


for cand_idx in 1:length(cand)
    for query_idx in 1:length(query)
        matrix[query_idx, cand_idx] = similarity(tracesim, query[query_idx], cand[cand_idx], df)
    end
end


# 237531, 0.8572043698222129
print(matrix)
print(aggregateMatrix("avg_query_cand", matrix))