using ArgParse
using util
using ranking
using preprocessing
using Method
using data
using Metrics
using preprocessing
using JSON
using DataStructures


# Include run_method file
dir_name, script_name = splitdir(@__FILE__)
include(joinpath(dir_name,"run_method.jl"))

s = ArgParseSettings()
@add_arg_table s begin
    "bug_dataset"
        help="File that contains all bug report data(categorical, summary,...)"
    "evaluation"
        help="Evaluation dataset"
    "method_name"
    "search_space_script"
    "--max_evals"
            help = ""
            arg_type = Int
            default = 100
    "--test"    
        help="Test dataset"
end

import_settings(s, get_basic_args())

parsed_args = parse_args(s, as_symbols=true)

@info parsed_args

# Load validation set. This set is used to find the best parameters.
validationDataset = readDataset(parsed_args[:evaluation])

# Load search space for the experiment
using PyCall

hyperopt = pyimport("hyperopt")
space_script_path = parsed_args[:search_space_script]

py"""
space_script = $(space_script_path)
f = open(space_script, 'r')
code_str = f.read()
f.close()

exec(compile(code_str, space_script, 'exec'))
"""

fixed_values = py"""fixed_values"""
space = py"""space"""


# Objective Function
function objective(params...)
    exp_parameter = Dict(parsed_args)

    for (k,v) in  params[1]
        exp_parameter[Symbol(k)] = v
    end

    for (k,v) in  fixed_values
        exp_parameter[Symbol(k)] = v
    end
    
    results = Dict()

    timeScore = @elapsed begin
        try
            results = run_method(validationDataset.bugIds, exp_parameter)
        catch e
            @error exception=(e, catch_backtrace())
            throw(e)
        end
    end

    map = 0.0
    auc = 0.0
    rr1 = 0.0
    rr5 = 0.0
    
    print(typeof(results))
    for r in results
        if r["label"] == "MAP_RecallRate"
            map = r["map"]::Float64
            rr1 = r["rr"][1]::Float64
            rr5 = r["rr"][5]::Float64
        elseif r["label"] == "BinaryPredictionROC"
            auc = r["auc"]::Float64
        end
    end



    loss = 2.0 - (map + auc)

    new_trial = OrderedDict(:x=> params[1],:loss => loss, :map=> map, :auc=>auc,:rr1 => rr1, :rr5=>rr5 , :args => exp_parameter)
    @info new_trial
    @info "Finish a new trial in $(timeScore)"

    return Dict("loss"=> 2.0 - (map + auc), "status"=> hyperopt.STATUS_OK, "results"=> results, "map"=> map, "auc"=> auc,
                "args"=> exp_parameter)
end


# Find the best parameters
trials = hyperopt.Trials()
best = hyperopt.fmin(objective,
            space=space,
            algo=hyperopt.tpe.suggest,
            max_evals=parsed_args[:max_evals],
            trials=trials)



@info "Paremeter search is over!"
@info "Best: $(best)"

@info "Sorted trial:"

all_trials = [ OrderedDict{Symbol, Any}(:vals => t["misc"]["vals"], :loss => t["result"]["loss"], :map => t["result"]["map"], :auc => t["result"]["auc"], :result => t["result"]["results"], :args => t["result"]["args"]) for t in trials]

sort!(all_trials, by=x-> x[:loss])

for t in all_trials
    @info "$(t)"
end


best_result = all_trials[1]
println(best_result)


if !isnothing(get(parsed_args, :test, nothing))
    @info "Start test:"
    testDataset = readDataset(parsed_args[:test])
    best_args = Dict{Symbol, Any}(( Symbol(k) => v for (k, v) in best_result[:args]))

    @info "$(best_args)"

    results = run_method(testDataset.bugIds, best_args)

    @info  JSON.json(results)
end
