module data

export readReportsFromJson, ReportDataset, readDataset, load_preselected_candidates, readDataset, PreselectedCandidates, load_training_pairs, load_negative_candidates

using DataStructures
using preprocessing
using util
using JSON
using Dates


function readReportsFromJson(filePath, preOpt::PreprocessingOption; stopId::Int64=0)
    l = Dict{Int64, Report}()
    vocab = Vocab(Dict{String, UInt32}(), Set())
    lastCreationts = -1
    reportJsons = JSON.parsefile(filePath)

    for reportJson in reportJsons
        reportId = reportJson["bug_id"]
        dup_id = reportJson["dup_id"]

        if isnothing(dup_id)
            dup_id = reportId
        end

        # println("Report: $(reportId)")
        
        curCreationts = reportJson["creation_ts"]
        creation_day_ts = calculateDayTimestamp(curCreationts)
        
        stacks = preprocessStacktrace(reportId, reportJson["stacktrace"], vocab, preOpt)
        l[reportId] = Report(reportId, dup_id, curCreationts, creation_day_ts, stacks)

        if stopId > 0
            if lastCreationts > curCreationts
                error("You cannot remove reports created after a report if the json is not sorted")
            end

            if stopId == reportId
                break
            end
        end

        lastCreationts = curCreationts
    end

    return (l,vocab)
end


struct ReportDataset
    info::String
    bugIds::Vector{Int64}
    duplicateIds::Vector{Int64}
end


function readDataset(file)
    open(file, "r") do io
        info = strip(readline(io))
        bugIds = [parse(Int64, id) for id in split(strip(readline(io)))]
        duplicateIds = [parse(Int64, id) for id in split(strip(readline(io)))]

        return ReportDataset(info, bugIds, duplicateIds)
    end
end

function load_training_pairs(path)
    pairs = MutableLinkedList{Tuple{Int64,Int64,Int64}}()

    open(path, "r") do io
        for line in eachline(io)
            split_line = split(strip(line), ",")
            push!(pairs, (parse(Int64, split_line[1]), parse(Int64, split_line[2]), parse(Int64, split_line[3])))
        end
    end

    return pairs
end

function load_negative_candidates(path)
    negative_candidates = Vector{Vector{Int64}}()

    open(path, "r") do io
        for line in eachline(io)
            split_line = split(strip(line), " ")

            v = Vector{Int64}(undef, length(split_line))

            for i=1:length(split_line)
                v[i] = parse(Int64, split_line[i])
            end
            
            push!(negative_candidates, v)
        end
    end

    return negative_candidates
end


struct PreselectedCandidates
    query_id::Int64
    candidates::Vector{Int64}
end


function load_preselected_candidates(path)
    preselectedList = MutableLinkedList{PreselectedCandidates}()

    open(path, "r") do io
        for line in eachline(io)
            split_line = split(strip(line), " ")

            v = Vector{Int64}(undef, length(split_line) - 1)

            for i=2:length(split_line)
                v[i-1] = parse(Int64, split_line[i])
            end

            push!(preselectedList, PreselectedCandidates(parse(Int64, split_line[1]), v))
        end
    end

    return preselectedList
end


end