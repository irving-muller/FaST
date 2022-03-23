using ArgParse
using util
using ranking
using preprocessing
using Method
using data
using Metrics
using preprocessing
using similarity_scorer
using JSON


# using Profile
# using ProfileSVG

# using InteractiveUtils



function get_method_args()
    s = ArgParseSettings()

    # The arguments of each method is defined here
    #TODO: find a  best solution
    
    @add_arg_table s begin
        # TraceSim and GreedyTraceSim
        "--alpha"
        arg_type=Float64
            default=1.0
            help="local weight parameter"
        "--beta"
        arg_type=Float64
            default=1.0
            help="global weight parameter"
        "--gamma"
        arg_type=Float64
            default=0.0
            help="position difference parameter"

        #TraceSim
        "--sigmoid"
            action = :store_true
            help="Use sigmoid fucntion to compute the global weight"
        "--b"
            arg_type=Float64
            default=1.0
            help="constant of sigmoid function"
        "--idf"
            action = :store_true
            help="Use idf instead of df"
        "--reciprocal_func"
            action = :store_true
            help="use reciprocal function instead of exponential"
        "--no_norm"
            action = :store_true
            help="Do not normalize output"
        "--sum_match"
            action = :store_true
            help="Sum match"

        # PDM
        "--c"
        arg_type=Float64
            default=1.0
            help="position"
        "--o"
        arg_type=Float64
            default=1.0
            help="offset"
            
        # TF-IDF
        "--tf_scheme"
            default="sqrt"
        "--idf_scheme"
            default="smooth"
        
        #BM25
        "--k1"
        arg_type=Float64
            default=1.0
            help="position"

        #TF-IDF 
        "--ngram"
            arg_type=Int64
            default=1
            help="position"
        "--add_query_freq"
            action = :store_true
            help=""
        #Durfex
        "--trim_level"
            default="function"

    end

    s.add_help = false

    return s

end

function run_method(queries::Vector{Int64}, parsed_args::Dict)
    preOpt = generatePreprocessingOption(max_depth=parsed_args[:max_depth], filter_recursion=parsed_args[:filter_recursion],  unique_ukn_report=!parsed_args[:keep_ukn], preprocess_func=get(parsed_args, :trim_level, "function"), select_interesting_stacks=parsed_args[:interesting_stacks])
    reportid2report, vocab = readReportsFromJson(parsed_args[:bug_dataset], preOpt, stopId=queries[end])

    # add ngrams to vocab
    ngram = get(parsed_args, :ngram, 1)
    
    if ngram > 1
        @info "Add ngrams to vocab. ngram=$(ngram)"
        add_ngram_to_vocab!(vocab, reportid2report, ngram)
    end

    @info "Prepocess dataset: $(parsed_args[:bug_dataset]).\tN of reports: $(length(reportid2report))\tVocab size: $(length(vocab.vocab))\tLast report: $(queries[end])"
    @info "Creating scorer and seting experiment up "
    

    update_doc_freq = get(parsed_args, :upd_doc_freq, true)
    docFreqRemoval = createDocFreqRemoval(parsed_args[:filter_func], parsed_args[:filter_func_k])
    scorer = createScorer(parsed_args[:method_name], parsed_args, update_doc_freq)
    strategy = initSunStrategy(queries, reportid2report, scorer, docFreqRemoval , vocab, parsed_args[:window], parsed_args[:freq_by_stacks], 
                            parsed_args[:static_df_ukn], update_doc_freq, get(parsed_args, :smooth, false))
    metricList = [createMAP_RecallRate(), createBinaryPredictionROC()]

    @info "Run evaluation"
    results = evaluate(strategy, queries, reportid2report::Dict{Int64, Report},  scorer, metricList)

    # Profile.clear()
    # @profile evaluate(strategy, queries[1:10], reportid2report::Dict{Int64, Report},  scorer, metricList)
    # ProfileSVG.save(joinpath("/home/irmul/workspace/bug_deduplication_stack_traces", "prof.svg"))

    return results
end

if (abspath(PROGRAM_FILE) == @__FILE__) 
    # || !isnothing(findfirst("run_debugger.jl", PROGRAM_FILE))
    basic_args = get_basic_args()
    s = ArgParseSettings()

    @add_arg_table s begin
        "bug_dataset"
            help="File that contains all bug report data(categorical, summary,...)"
        "evaluation"
            help="Evaluation dataset with pairs"
        "method_name"
            help="Method name. Options: trace_sim, our_method"
    end

    import_settings(s, get_method_args())
    import_settings(s, basic_args)

    parsed_args = parse_args(s, as_symbols=true)

    @info parsed_args

    validationDataset = readDataset(parsed_args[:evaluation])
    results = run_method(validationDataset.bugIds, parsed_args)


    @info JSON.json(results)
end



