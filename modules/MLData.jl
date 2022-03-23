module MLData

export generate_pairs, generate_ftrs


using ranking
using similarity_scorer
using DataStructures
using util
using StatsBase
using Random




###############################################################
# Generate Pairs
###############################################################

function add_neg_pair!(neg_pairs_set::Set{Tuple{Int64, Int64, Int64}}, query::Report, cand::Report)
    previous_len= length(neg_pairs_set)

    if query.id > cand.id
        push!(neg_pairs_set, (cand.id, query.id, 0))
    else
        push!(neg_pairs_set, (query.id, cand.id, 0))
    end

    return length(neg_pairs_set) != previous_len
end

function generate_pairs(duplicate_ids, reportid2report, dupid2bucket::Dict{Int64, Bucket}, index::Dict{UInt32, MutableLinkedList{PositionPosting}}, rate=1.0)
    pairs = MutableLinkedList{Tuple{Int64, Int64, Int64}}()
    selected_buckets = Set(map(dup_id -> dupid2bucket[reportid2report[dup_id].dup_id], duplicate_ids))

    all_dup_reports = Vector{Report}()
    sizehint!(all_dup_reports, length(reportid2report))

    # Generate positive pairs
    for bucket in selected_buckets
        p = 0
        for (i, query) in enumerate(bucket.reports)
            push!(all_dup_reports, query)

            for (j, cand) in enumerate(bucket.reports)
                i >= j && continue
                push!(pairs, (query.id, cand.id, 1))
                p+=1
            end
        end
    end

    @info("Generate $(length(pairs)) of positive pairs.")


    # Generate negatives pairs
    n_neg_pair_per_query = Int64(ceil(length(pairs)/length(all_dup_reports)))
    buckets_it = Iterators.cycle(values(dupid2bucket))
    neg_pairs_set = Set{Tuple{Int64, Int64, Int64}}()

    @info("Number of negative pair per query: $(n_neg_pair_per_query)")

    n_negative_pairs = 0
    n_random_pairs = 0

    for query in all_dup_reports
        # First, try to generate negative pairs whose query and candidate contain shared functions
        neg_candidates = Vector{Report}()
        function_ids = Set{Int64}(Iterators.flatten(query.stacks))
    
        for func_id in function_ids
            append!(neg_candidates, map(x -> x.report, filter(posting-> posting.report.dup_id != query.dup_id, index[func_id])))
        end

        n_previous_pairs = length(neg_pairs_set)

        shuffle!(neg_candidates)

        for cand in neg_candidates
            cand.dup_id == query.dup_id && throw("$((query.id, cand.id, 0)) is not correct. The reports are from the same bucket")

            add_neg_pair!(neg_pairs_set, query, cand)

            # Break loop with the number of generated negative pairs per query was achieved
            length(neg_pairs_set) - n_previous_pairs == n_neg_pair_per_query && break
        end

        n_negative_pairs+=length(neg_pairs_set) - n_previous_pairs

        # If it was not generated enough number of negative pairs, then generate randomly the negative pairs.
        n_neg_tmp = length(neg_pairs_set)

        for bucket in buckets_it
            # Break loop with the number of generated negative pairs per query was achieved
            length(neg_pairs_set) - n_previous_pairs == n_neg_pair_per_query && break

            bucket.master.id == query.id && continue
            rand() > 0.5 && continue

            idx = rand(1:length(bucket.reports))

            for  (i, cand) in enumerate(bucket.reports)
                i != idx && continue
                
                add_neg_pair!(neg_pairs_set, query, cand)
                break
            end
        end

        n_random_pairs+= length(neg_pairs_set) - n_neg_tmp
    end

    @info("Generate $(length(neg_pairs_set)) of negative pairs. (selected: $(n_negative_pairs), random: $(n_random_pairs))")

    for np in neg_pairs_set
        push!(pairs, np)
    end

    return pairs
end


function generate_ftrs(pairs, reportid2report, df_vec, pos_scaler)
    reportid2bowpos = Dict{Int64, Dict{UInt32, Vector{Int16}}}()

    X = Vector{Vector{Vector{Float32}}}(undef, length(pairs))
    report_lengths = Vector{Int64}(undef, length(pairs))
    y = Vector{Int64}(undef, length(pairs))

    sizehint!(X, length(pairs))

    for (idx, pair) in enumerate(pairs)
        ftr_list = Vector{Vector{Float32}}()
        q_id = pair[1]
        c_id = pair[2]

        if !haskey(reportid2bowpos, q_id)
            reportid2bowpos[q_id] = Dict{UInt32, Vector{Int16}}(aggregatePositionByFunction(reportid2report[q_id]))
        end

        if !haskey(reportid2bowpos, c_id)
            reportid2bowpos[c_id] =  Dict{UInt32, Vector{Int16}}(aggregatePositionByFunction(reportid2report[c_id]))
        end

        qfunc2bowpos = reportid2bowpos[q_id]
        cfunc2bowpos = reportid2bowpos[c_id]

        func_set  = Set(keys(qfunc2bowpos))
        union!(func_set, keys(cfunc2bowpos))

        empty = Vector{Int16}()

        sum_rep_len = 0
        for func_id in func_set
            q_bow_pos = get(qfunc2bowpos, func_id, empty)
            c_bow_pos = get(cfunc2bowpos, func_id, empty)
            min_length = min(length(q_bow_pos), length(c_bow_pos))

            for i=1:min_length
                ftr = Vector{Float32}(undef, 3)

                ftr[1] = df_vec[func_id]
                ftr[2] = q_bow_pos[i]/pos_scaler
                ftr[3] = c_bow_pos[i]/pos_scaler
                push!(ftr_list, ftr)
                sum_rep_len+=2
            end

            for i=min_length+1:length(q_bow_pos)
                ftr = Vector{Float32}(undef, 3)

                ftr[1] = df_vec[func_id]
                ftr[2] = q_bow_pos[i]/pos_scaler
                ftr[3] = -1.0/pos_scaler
                push!(ftr_list, ftr)
                sum_rep_len+=1
            end

            for i=min_length+1:length(c_bow_pos)
                ftr = Vector{Float32}(undef, 3)

                ftr[1] = df_vec[func_id]
                ftr[2] = c_bow_pos[i]/pos_scaler
                ftr[3] = -1.0/pos_scaler
                push!(ftr_list, ftr)
                sum_rep_len+=1
            end
        end

        X[idx] = ftr_list
        y[idx] = pair[3]

        report_lengths[idx] = sum_rep_len

        # println("#### $((q_id, c_id))")
        # println(qfunc2bowpos)
        # println(cfunc2bowpos)
        # println(ftr_list)
        # println(sum_rep_len)
        # println(pair[3])


    end

    return (X,report_lengths, y)
end

end