using ArgParse
using JSON
using DataStructures



function main()
    s = ArgParseSettings()

    @add_arg_table s begin
        "dir"
            help="path to directory that contains the log files"
        "pattern"
            help="regex"

        "--n_chunks"
            default=50
            arg_type=Int64
    end

    args = parse_args(s, as_symbols=true)


    runs = Dict{Int64, Dict{Symbol, Any}}()


    n_chunks = args[:n_chunks]

    @info "Dir: $(args[:dir])"
    @info "Pattern: $(args[:pattern])"
    @info "Num of chunks: $(n_chunks)"

    pattern = Regex(args[:pattern])

    dataset_name = ""
    validation_name = ""
    test_name = ""
    space_script = ""
    method_name = ""

    for file_name in readdir(args[:dir])
        file_path = joinpath(args[:dir], file_name)

        !isfile(file_path) && continue
        !occursin(pattern, file_name) && continue

        info =  Dict{Symbol, Any}()

        info[:results_exec] = Vector()
        info[:validation_results] = Vector()

        @info file_name
        

        job_id = nothing

        open(file_path, "r") do file
            for l in eachline(file)
                l = replace(l, "[ Info: " => "")
                l = replace(l, r"^.*best +loss: +[0-9.]+\]" => "")

                o = nothing
                
                try
                    o = eval(Meta.parse(l))

                    if haskey(o, :x)
                        push!(info[:results_exec], o)
                        new_job_id = parse(Int64, match(r"validation_chunk_([0-9]+).txt", o[:args][:evaluation])[1])

                        if !isnothing(job_id) && job_id != new_job_id
                            @error("Different chunk ids!!")
                        end
                        job_id = new_job_id
                    end

                    if haskey(o, :vals)
                        push!(info[:validation_results], o)
                    end

                    if haskey(o, :evaluation)
                        info[:best_par] = o
                        
                        if length(dataset_name) > 0 &&  dataset_name != o[:bug_dataset] 
                            @error("Dataset is different")
                        end

                        validation_nm_tmp = replace(o[:evaluation], r"[0-9]+.txt" => ".txt")

                        if length(validation_name) > 0 &&  validation_name != validation_nm_tmp
                            @error("Validation is different")
                        end

                        test_nm_tmp = replace(o[:test], r"[0-9]+.txt" => ".txt")
                        if length(test_name) > 0 &&  test_name != test_nm_tmp
                            @error("Test is different")
                        end
                        
                        if length(space_script) > 0 &&  space_script != o[:search_space_script]
                            @error("space_script is different")
                        end
                        
                        if length(method_name) > 0 &&  method_name != o[:method_name]
                            @error("method is different")
                        end

                        dataset_name = o[:bug_dataset]
                        validation_name = validation_nm_tmp
                        test_name = test_nm_tmp
                        space_script = o[:search_space_script]
                        method_name = o[:method_name]
                    end

                    continue
                catch
                end    


                try
                    o = JSON.parse(l)
                    
                    if isa(o, Vector)
                        info[:test_results] = o
                    end
                catch
                end            
            end
        end

        isnothing(job_id) && continue

        runs[job_id] = info
    end

    missing_ids = []
    best_parameters = Vector{Any}([nothing for i=1:n_chunks])


    test_results = OrderedDict("map"=>OrderedDict{Int64,Float64}(), "auc"=> OrderedDict{Int64,Float64}())

    for rr_k in 1:20
        test_results["rr@$(rr_k)"] = Dict{Int64,Float64}()
    end


    for i=0:n_chunks-1
        if !haskey(runs, i)
            push!(missing_ids, i)
            continue
        end

        validation_results = runs[i][:validation_results]

        if length(validation_results) == 0
            push!(missing_ids, i)
            continue
        end

        sort!(validation_results, by= j->j[:loss])
        best_parameters[i+1] = validation_results[1][:args]


        if !haskey(runs[i], :test_results) 
            push!(missing_ids, i)
            continue
        end

        for r in runs[i][:test_results]
            if r["label"] == "MAP_RecallRate"
                test_results["map"][i] = r["map"]


                for (k,v) in r["rr"]
                    test_results["rr@$(k)"][i] = v
                end
            elseif r["label"] == "BinaryPredictionROC"
                test_results["auc"][i] = r["auc"]
            end
        end
    end


    @info [ i-1=>best_parameters[i] for i=1:length(best_parameters)]

    sort!(missing_ids)

    if length(missing_ids) > 0
        @info "Missing chunks (test): $(missing_ids)"

        println("Start commands:")
        for chunk_id in missing_ids 

            if !isdefined(best_parameters, chunk_id+1)
                println("Chunk $(chunk_id) is undefined")

                continue
            end
            
            chunk_param = best_parameters[chunk_id+1]

            command = "julia -Jimage.so experiments/run_method.py $(chunk_param[:bug_dataset]) $(chunk_param[:test]) $(chunk_param[:method_name])"

            for (k, v) in chunk_param
                if k ∈ [:method_name, :bug_dataset, :evaluation, :test, :search_space_script, :max_evals]
                    continue
                end

                isnothing(v) && continue
                
                if isa(v, Bool)
                    if v
                        command = command * " --$(k)"
                    end
                else
                    command = command * " --$(k) $(v)"
                end

            end
            println(command)
        end
    end

    @info "$(method_name)"
    @info "$(dataset_name)\t$(validation_name)\t$(test_name)\t$(space_script)"

    @info JSON.json(test_results)
end

main()

