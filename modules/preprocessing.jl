module preprocessing

export PreprocessingOption, generatePreprocessingOption, preprocessFunction, preprocessPackage!, stdFunctionPreprocess!, preprocessStacktrace, retrieveDocfreq!, DocFreqRemoval, removeCommonFunction!, createDocFreqRemoval

using SHA
using DataStructures
using util

struct Frame
    func::String
    is_crash::Bool # Only for Gnome True if this is the frame where we crashed. For example, in a GDB trace, is_crash true if this is the frame where the signal handler was called.
end

Frame(func::String) = Frame(func, false)


function createFrame(frameJson)
    # Consider frame without 
    func_name = get(frameJson, "function", nothing)
    func_name =  isnothing(func_name) ? ""  :  string(func_name)

    is_crash = get(frameJson, "is_crash", "0") == "1"
    if !is_crash && occursin("<signal handler called>", func_name)
        # println("$(func_name) should be is_crash")
        is_crash = true
    end

    return Frame(func_name, is_crash)
end

struct PreprocessingOption
    max_depth::Int32
    filter_recursion::String
    ukn_tkn::String
    unique_ukn_report::Bool
    trim_len::Int32
    rm_sub_stacks::Bool
    select_interesting_stacks::Bool
end

function generatePreprocessingOption(;max_depth=300, filter_recursion="none", ukn_tkn="??", unique_ukn_report=true,
    preprocess_func="function", rm_sub_stacks=true, select_interesting_stacks=false)

    trim_len = 0

    if preprocess_func == "function"
        trim_len = 0
    elseif preprocess_func == "class"
        trim_len = 1
    elseif preprocess_func == "package"
        trim_len = 2
    end

    @info "Trim length: $(trim_len)"

    if select_interesting_stacks
        @warn "select_interesting_stacks only works for Gnome!"
    end
    
    return PreprocessingOption(max_depth, filter_recursion, ukn_tkn, unique_ukn_report, trim_len, rm_sub_stacks, select_interesting_stacks)
end



function preprocessFunction(func_name::String)::String
    # Strip __GI__ in the begining of function call
    func_name = replace(func_name, r"\(.*\)" => "")
    # Strip _ in the begining of function call
    func_name = replace(func_name, r"^_*GI_+" => "")
    # Remove underscore at the beggining of the function
    func_name = replace(func_name, r"^_+" => "")

    # todo: Add S3M preprocessing: remove numbers and punctuations(except , and .) Do not remove functions that only contain numbers 
    # todo: transform to lower character as S3M

    # Java anonymous function and inner function
    func_name = replace(func_name, r"[$][0-9]+" => "\$")   
    func_name = replace(func_name, r"invoke[0-9]+" => "invoke")   

    # func_name = replace(func_name, r"(?<=[a-zA-Z])[0-9]+$" => "" )
    # func_name = replace(func_name, r"[^a-zA-Z.,_<>$]"=> "" )

    return strip(func_name)
end


function trim_function_name(func_name, trim_len, sep='.')
    if trim_len == 0
        return func_name
    end 

    last_idx = 0
    ndots = 0

    for idx in Iterators.reverse(eachindex(func_name))
        if func_name[idx] === sep
            ndots+=1

            if ndots == trim_len
                last_idx = idx - 1
                break
            end
        end
    end

    if ndots != trim_len 
        return ""
    end
    
    return func_name[1:last_idx]
end

function preprocess_stack!(vocab::Vocab, frames::MutableLinkedList{Frame}, ukn_tkn, ukn_id, trim_len, max_depth=300)::Vector{UInt32}
    token_id::UInt32 = 0
    vecLen = length(frames) > max_depth ? max_depth : length(frames)
    function_ids = Vector{Int32}()
    sizehint!(function_ids, length(vecLen))

    for (depth, frame) in enumerate(frames)
        depth > max_depth && break

        if length(frame.func) == 0 || frame.func == ukn_tkn
            token_id = ukn_id
        else
            func_name = trim_function_name(frame.func, trim_len)

            if isempty(func_name)
                continue
            end

            func_name = preprocessFunction(frame.func)
            token_id = setdefault!(vocab, func_name)
        end

        push!(function_ids, token_id)
    end

    if length(function_ids) == 0
        @warn "Stack trace is empty. Fill it with a fake token."
        token_id = setdefault!(vocab, "@@@$(length(vocab.vocab))@@@")
        push!(function_ids, token_id)
    end

    return function_ids
end


function stdFunctionPreprocess!(vocab::Vocab, frames::MutableLinkedList{Frame}, ukn_tkn, ukn_id, max_depth=300)::Vector{UInt32}
    vecLen = length(frames) > max_depth ? max_depth : length(frames)

    if length(frames) == 0
        @warn "Stack trace is empty. Fill it with a fake token."
        function_ids = Vector{UInt32}(1)    
        function_ids[1] = setdefault!(vocab, "@@@$(length(vocab.vocab))@@@")
        return function_ids
    end

    function_ids = Vector{UInt32}(undef, vecLen)
    token_id::UInt32 = 0

    for (depth, frame) in enumerate(frames)
        depth > max_depth && break

        if length(frame.func) == 0 || frame.func == ukn_tkn
            token_id = ukn_id
        else
            func_name = preprocessFunction(frame.func)
            token_id = setdefault!(vocab, func_name)
        end
        
        function_ids[depth] = token_id
    end

    return function_ids
end



# vocab = Dict{String, UInt32}()
# preprocessPackage!(vocab, [Dict("function"=> "__gi__func()"),Dict("function"=> "org.eclipse.package.class.method"),Dict("function"=> 1)])
# vocab = Dict{String, UInt32}("UKN" => 0)
# stdFunctionPreprocess(vocab, [Dict("function"=> "__GI__func()"),Dict("function"=> "UKN"),Dict("function"=> 1)], "UKN", 0)


"""
    rm_sub_stack!(stacktraces::Vector{Vector{UInt32}})

    Remove stacks that are sub sequence of other ones. We consider if st1 is subsequence st2 when 
        the prefix of st2 is equal to st1, i.e., st1[1] ==st2[1], st1[2] ==st2[2] .. st1[n] ==st2[n] where n is st1 length.
"""
# function rm_sub_stack(stacktraces::Vector{Vector{UInt32}})::Vector{Vector{UInt32}}
function rm_sub_stack!(stacktraces::Vector{Vector{UInt32}})::Vector{Vector{UInt32}}
    length(stacktraces) == 1 && return stacktraces
    idx_to_rm = Vector{Int64}()

    sorted_stacktraces = sort!(stacktraces, by=v -> length(v))

    for (i, stack1) in enumerate(sorted_stacktraces)
        stack1_len = length(stack1)

        for j = i + 1:length(sorted_stacktraces)
            stack2 = sorted_stacktraces[j]

            stack1_len > length(stack2) && continue

            # Prefix of st2 is equal to st1
            frame_idx = 1

            while frame_idx != stack1_len + 1
                stack2[frame_idx] != stack1[frame_idx] && break
                frame_idx += 1
            end

            if frame_idx - 1 === stack1_len
                push!(idx_to_rm, i)
                break
            end
        end
    end

    # if length(idx_to_rm) > 0
    #     println(idx_to_rm)
    #     println(stacktraces)
    # end

    length(idx_to_rm) > 0 && deleteat!(sorted_stacktraces, idx_to_rm)

    return sorted_stacktraces
end



function rmDuplicateStacks(stacktraces::Vector{Vector{UInt32}})::Vector{Vector{UInt32}}
    hash2stack = Dict{String,Vector{UInt32}}([(bytes2hex(sha2_512(string(stack))), stack) for (stack_idx,  stack) in enumerate(stacktraces)])
    
    if length(hash2stack) == length(stacktraces)
        return stacktraces
    end

    return [stack for stack in values(hash2stack)]
end

# println(rmDuplicateStacks([[0x00001,0x00002,0x00003],[0x00002,0x00003,0x00004,0x00005],[0x00003,0x00003,0x00003,0x00003]]))
# println(rmDuplicateStacks([[0x00001,0x00002,0x00003],[0x00001,0x00002,0x00003],[0x00003,0x00003,0x00003,0x00003]]))

function validateStacktraceFrames!(mainVector, stack, report_id, ukn_tkn,  is_nested)
    new_stacktrace = MutableLinkedList{Frame}()
    last_depth = -1

    for frameJson in stack["frames"]
        depth = typeof(frameJson["depth"]) == Int64 ? frameJson["depth"] : parse(Int64, frameJson["depth"])
        pos_diff = depth - last_depth - 1
        
        frame = createFrame(frameJson)

        if pos_diff > 0
            # Fix stack trace
            for i in 1:pos_diff
                # It is missing frames from last_depth until depth
                # missing_depth = last_depth + i
                push!(new_stacktrace, Frame(ukn_tkn))    
        end
            
            # @warn "report stacktrace $(report_id) is missing frames."
        elseif pos_diff < 0
            # There are multiple stacks in the stack variable
            if length(new_stacktrace) > 0
                push!(mainVector, new_stacktrace)
            end

            last_depth = depth
            new_stacktrace = MutableLinkedList{Frame}()

            for i in 0:depth - 1
                push!(new_stacktrace, Frame(ukn_tkn))
            end
            # logging.getLogger().error("More than one stack in report {}.".format(report_id, stacktrace['frames']))
        end

        last_depth = depth
        append!(new_stacktrace, frame)
    end


    if length(new_stacktrace) == 0 && !is_nested
        @warn "preprocessed_frame is empty for report $(report_id)"
        return 
    end

    push!(mainVector, new_stacktrace)
end

function validateStacktrace!(main_frames, stack, report_id, ukn_tkn)
    validateStacktraceFrames!(main_frames, stack, report_id, ukn_tkn, false)

    nested = get(stack, "nested", nothing)
    
    if !isnothing(nested) && length(stack["nested"]) > 0
        validateStacktraceFrames!(main_frames, stack["nested"][1], report_id, ukn_tkn, true)    
    end
end


function removeRecursiveCalls(frames::Vector{UInt32}, ukn_function::UInt32, recursion_removal::String)::Vector{UInt32}
    if recursion_removal == "none"
        return frames
    end

    clean_stack = MutableLinkedList{UInt32}()

    if recursion_removal == "brodie"
        # Quickly Finding Known Software Problems via Automated Symptom Matching - Mark Brodie 2006
        previous_function = frames[1]
        push!(clean_stack, frames[1])
        
        for (idx, fr) in enumerate(frames)
            idx == 1 && continue

            if fr == ukn_function || fr != previous_function
                push!(clean_stack, fr)
            end

            previous_function = fr
        end
    elseif recursion_removal == "modani"
        # Automatically Identifying Known Software Problems - Natwar Modani 2007
        idx = 1

        while idx < length(frames) + 1
            fr = frames[idx]
            end_idx = -9999999

            push!(clean_stack, fr)

            if fr != ukn_function
                for bk_idx in (length(frames)):-1:idx + 1
                    bk_func = frames[bk_idx]

                    if fr == bk_func
                        end_idx = bk_idx
                        break
                    end

                end
            end

            if end_idx > 0
                idx = end_idx + 1
            else
                idx += 1
            end
        end
    else
        error("Invalid argument value for recursion_removal: $(recursion_removal)")
    end

    if length(clean_stack) != length(frames)
        return collect(clean_stack)
    else
        return frames
    end
end

function extract_interesting_stacks(stacktraceLinkedList::MutableLinkedList{MutableLinkedList{Frame}})::MutableLinkedList{MutableLinkedList{Frame}}
    # Following: https://bazaar.launchpad.net/~bgo-maintainers/bugzilla-traceparser/3.4/view/head:/lib/TraceParser/Trace.pm line 256
    length(stacktraceLinkedList) == 1 && return stacktraceLinkedList

    interesting_stacks = MutableLinkedList{MutableLinkedList{Frame}}()

    for stack in stacktraceLinkedList
        for frame in stack
            if frame.is_crash
                push!(interesting_stacks, stack)
                break
            end
        end
    end

    length(interesting_stacks)  > 0 && return interesting_stacks

    # Search for threads that have a function with
    # "signal" or "segv" in the name.


    # If a trace lacks <signal handler called>, we determine the
    # "interesting thread" by looking for a thread that has
    # functions that match this regex.
    possible_crash_functions = r"signal|segv|sighandler"i
    for stack in stacktraceLinkedList
            for frame in stack
                if occursin(possible_crash_functions, frame.func)
                    push!(interesting_stacks, stack)
                    break
            end
        end
    end

    length(interesting_stacks)  > 0 && return interesting_stacks


    # If we still don't have a thread, return every first thread whose
    # last function isn't some form of wait or one of the ignored
    # functions.


    # Or if that fails, by a thread whose last function *doesn't* match
    # this regex.
    wait_function =  r"wait|sleep|poll"i

    # However, some wait functions are interesting--for example,
    # if we're waiting on a lock, that's interesting during
    # deadlock traces.
    interesting_wait_function = r"lock"i;
    ignore_functions = Set(["kernel_vsyscall", "libc_start_main",  "raise", "abort", "poll", "??"])

    for stack in stacktraceLinkedList
        first_func = stack[1].func
        if ((!occursin(wait_function, first_func) || occursin(interesting_wait_function, first_func)) && first_func ∉ ignore_functions)
            push!(interesting_stacks, stack)
        end
    end

    length(interesting_stacks)  > 0 && return interesting_stacks

    return stacktraceLinkedList     
end

 

function preprocessStacktrace(report_id, stacktracesJson, vocab::Vocab,  opt::PreprocessingOption)
    ukn_tkn = opt.ukn_tkn

    if opt.unique_ukn_report
        # Create token for unknown function name. Give a unique id to the unknown values in stacktrace
        # Following ABRT, we only compare the function names and consider two unknown function(??) as different
        ukn_tkn = "$(ukn_tkn)$(report_id)"

        haskey(vocab.vocab, ukn_tkn) &&  error("Token for unknown function name in the report $(report_id) already exists.")
    end

    ukn_id = setdefault!(vocab, ukn_tkn, true)

    stacktraceLinkedList = MutableLinkedList{MutableLinkedList{Frame}}()

    if stacktracesJson isa Vector
        for stack in stacktracesJson
            validateStacktrace!(stacktraceLinkedList, stack, report_id, ukn_tkn)
        end
    else
        validateStacktrace!(stacktraceLinkedList, stacktracesJson, report_id, ukn_tkn)
    end

    if opt.select_interesting_stacks
        stacktraceLinkedList = extract_interesting_stacks(stacktraceLinkedList)
    end

    finalStacktraces = collect(stacktraceLinkedList)
    
    preprocessedStacktraces = Vector{Vector{UInt32}}(undef, length(finalStacktraces))
    
    for (idx, stack) in enumerate(finalStacktraces)
        preprocessedStacktraces[idx] = preprocess_stack!(vocab, stack, ukn_tkn, ukn_id, opt.trim_len, opt.max_depth)
        preprocessedStacktraces[idx] = removeRecursiveCalls(preprocessedStacktraces[idx], ukn_id, opt.filter_recursion)
    end

    if opt.rm_sub_stacks
        preprocessedStacktraces = rm_sub_stack!(preprocessedStacktraces)
        # preprocessedStacktraces = rmDuplicateStacks(preprocessedStacktraces)
    end


    length(preprocessedStacktraces) == 0 && error("$(report_id) contains 0 zero stacktraces".format(report_id))

    return preprocessedStacktraces  
end


abstract type CommonFunctionRemoval end

struct DocFreqRemoval <: CommonFunctionRemoval
    k::Float64 # threshold
    trim::Bool
end

function createDocFreqRemoval(name, k)
    if name == "none"
        return DocFreqRemoval(-1.0, false)
    elseif name == "threshold"
        return DocFreqRemoval(k / 100.00, false)
    elseif name == "threshold_trim"
        return DocFreqRemoval(k / 100.00, true)
    end
end

isToRemove(funcId, removalStrategy, docFreq) = ((docFreq.doc_freq[funcId] > removalStrategy.k * docFreq.nDocs) && (!docFreq.static_df_ukn || funcId ∉ docFreq.vocab.ukn_set))

function removeCommonFunction!(report::Report, removalStrategy::DocFreqRemoval, docFreq::DocFreq)
    if removalStrategy.k <= 0.0
        return 
    end

    idxs_to_remove_by_stack = Vector{BitVector}(undef, length(report.stacks))
    n_empty_stacks = 0
    empty_vectors = falses(length(report.stacks))
    
    for (stack_idx, stacktrace) in enumerate(report.stacks)
        idxs_to_remove =  get_indices_to_remove(stacktrace, removalStrategy, docFreq)
        idxs_to_remove_by_stack[stack_idx] = idxs_to_remove

        if sum(idxs_to_remove) == length(stacktrace)
            n_empty_stacks += 1
            empty_vectors[stack_idx] = true
        end
    end

    if n_empty_stacks == length(report.stacks)
    # If all stacks will be empty, so do not remove the common functions
        return
    end

    for stack_idx = 1:length(report.stacks)
        if empty_vectors[stack_idx]
            continue
        end

        deleteat!(report.stacks[stack_idx], idxs_to_remove_by_stack[stack_idx])
    end

    deleteat!(report.stacks, empty_vectors)
end

function get_indices_to_remove(stack::Vector{UInt32}, removalStrategy::DocFreqRemoval, docFreq::DocFreq)
    idxToRemove = falses(length(stack))

    if removalStrategy.trim
        for (idx, funcId) in enumerate(stack)
            (idx > 1 && !idxToRemove[idx - 1]) && break

            
            idxToRemove[idx] = isToRemove(funcId, removalStrategy, docFreq)
        end

        for (idx, funcId) in Iterators.reverse(enumerate(stack))
            (idx < length(stack) && !idxToRemove[idx + 1]) && break
        
            idxToRemove[idx] = isToRemove(funcId, removalStrategy, docFreq)
        end
    else
        for (idx, funcId) in enumerate(stack)
           idxToRemove[idx] = isToRemove(funcId, removalStrategy, docFreq)
        end
    end

    return idxToRemove
    # deleteat!(stack, idxToRemove)
end





end