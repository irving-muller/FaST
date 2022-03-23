using Test
using DataStructures


using preprocessing
using JSON
using util

# Test common function removal
k = 5.0
docFreqRemoval = createDocFreqRemoval("threshold", k)

docFreq = DocFreq([5.0,10.0,3.0,4.0,50.0,60.0,80.0], 100.0, true, false, Vocab(Dict{String, UInt32}(), Set{UInt32}()))
r = Report(1,1,0,0,[[1,2,5,5,5,2,6,6,3,4,7,1,7,7], [1,2,3], [7,6,5,5,6,7]])

removeCommonFunction!(r, docFreqRemoval, docFreq)

@test r.stacks == [[1,3,4,1],[1,3]]

r = Report(1,1,0,0,[[7,2,5,7], [2,2,2], [7,6,5,5,6,7]])
removeCommonFunction!(r, docFreqRemoval, docFreq)
@test r.stacks == [[7,2,5,7], [2,2,2], [7,6,5,5,6,7]]


docFreqRemoval = createDocFreqRemoval("threshold_trim", k)
r = Report(1,1,0,0,[[1,2,5,5,5,2,6,6,3,4,7,1,7,7], [1,2,3], [7,6,5,5,6,7], [2,2,2,2,1,6,6], [7,7,1,5,7,6,1]])
removeCommonFunction!(r, docFreqRemoval, docFreq)

@test r.stacks == [[1,2,5,5,5,2,6,6,3,4,7,1],[1,2,3], [1], [1,5,7,6,1]]


r = Report(1,1,0,0,[[7,2,5,7], [2,2,2], [7,6,5,5,6,7]])
removeCommonFunction!(r, docFreqRemoval, docFreq)

@test r.stacks == [[7,2,5,7], [2,2,2], [7,6,5,5,6,7]]

docFreqRemoval = createDocFreqRemoval("threshold", k)

docFreq = DocFreq([5.0,10.0,3.0,4.0,50.0,60.0,80.0], 100.0, true, true, Vocab(Dict{String, UInt32}(), Set{UInt32}([2])))
r = Report(1,1,0,0,[[1,2,5,5,5,2,6,6,3,4,7,1,7,7], [1,2,3], [7,6,5,5,6,7]])

removeCommonFunction!(r, docFreqRemoval, docFreq)

@test r.stacks == [[1,2,2,3,4,1],[1,2,3]]














# @testset "CreateFrame" begin
#   # CreateFrame
#   @test preprocessing_stacks.createFrame(Dict("function" => "test")).func == "test"
#   @test preprocessing_stacks.createFrame(Dict("function" => "")).func == ""
#   @test preprocessing_stacks.createFrame(Dict("function" => nothing)).func == ""
#   @test preprocessing_stacks.createFrame(Dict()).func == ""

#   @test preprocessing_stacks.preprocessFunction("__GI__libc_free (mem=0x3)") == "libc_free"
#   @test preprocessing_stacks.preprocessFunction("_GI_libc_free (mem=0x3)") == "libc_free"
#   @test preprocessing_stacks.preprocessFunction("__libc_free") == "libc_free"

#   vocab = Dict{String, UInt32}()
#   frames = MutableLinkedList{Frame}(Frame("org.eclipse.vcm.internal.core.ccvs.client.Connection.readLineOrUntil"), Frame("home.com"), Frame("__GI__libc_free"), Frame(""), Frame("org.eclipse.vcm"), Frame("1"), Frame("home.com"), Frame("UKN"))

#   @test preprocessing_stacks.preprocessPackage!(vocab, frames, nothing, nothing) == [0x00001, 0x00002, 0x00003]
#   @test preprocessing_stacks.preprocessPackage!(vocab, frames, nothing, nothing, 1) == [0x00001]

#   vocab = Dict{String, UInt32}("UKN"=>1)
  

#   @test preprocessing_stacks.stdFunctionPreprocess!(vocab, frames, "UKN", 1) == [0x00002, 0x00003, 0x00004,0x00001,0x00005,0x00006,0x00003, 0x00001]
#   @test preprocessing_stacks.stdFunctionPreprocess!(vocab, frames, "UKN", 1, 3) == [0x00002, 0x00003, 0x00004]


#   frames2 = MutableLinkedList{Frame}(Frame("org.eclipse"), Frame("home.com"), Frame("__GI__libc_free"))
#   frames3 = MutableLinkedList{Frame}(Frame("org.eclipse"), Frame("home.com"), Frame("__GI__libc_free"))
#   frames4 = MutableLinkedList{Frame}(Frame("org.eclipse.vcm.internal.core.ccvs.client.Connection.readLineOrUntil"), Frame("home.com"), Frame("__GI__libc_free"), Frame(""), Frame("org.eclipse.vcm"), Frame("1"), Frame("home.com"), Frame("UKN"))

#   l = preprocessing_stacks.rmDuplicateStacks([frames, frames2, frames4, frames3])
#   i = 0

#   for k in l
#     if k == frames2 || k == frames3
#       i+=1
#     elseif k == frames || k == frames4
#       i+=1
#     end
#   end
  
#   @test length(l) == 2 && i == 2


#   b = """
#   [
#     {
#      "bug_id": 1,
#      "stacktrace": [ 
#           {
#           "frames": [{"depth": 0, "function": "test1"}, {"depth": 1, "function": "__GI__libc_free"}, {"depth": 3, "function": "__GI__libc_free"}]
#           }  
#         ]      
#     },
#     {
#       "bug_id": 2,  
#     "stacktrace": [ 
#           {
#           "frames": [{"depth": 0, "function": "test1"}, {"depth": 3, "function": "__GI__libc_free"}, {"depth": 10, "function": "__GI__libc_free"}]
#           }  
#         ]      
#     },
#     {
#       "bug_id": 3,  
#       "stacktrace": [ 
#           {
#           "frames": [{"depth": 0, "function": "test1"}, {"depth": 1, "function": "__GI__libc_free"}, {"depth": 2, "function": "__GI__libc_free"}, {"depth": 2, "function": "__GI__libc_free"}]
#           }  
#         ]      
#     },
#     {
#       "bug_id": 4,  
#       "stacktrace": [ 
#           {
#           "frames": [{"depth": 0, "function": "test1"}, {"depth": 1, "function": "test1"}, {"depth": 2, "function": "test2"}, {"depth": 2, "function": "test1"}]
#           }  
#         ]      
#     },
#     {
#       "bug_id": 5,  
#       "stacktrace":
#           {
#           "frames": [{"depth": 0, "function": "test1"}, {"depth": 1, "function": "test3"}, {"depth": 2, "function": "test2"}, {"depth": 2, "function": "test1"}]
#           }  
#     }
#     ,
#     {
#       "bug_id": 6,  
#       "stacktrace":[
#           {
#           "frames": [{"depth": 0, "function": "test1"}, {"depth": 1, "function": "test3"}, {"depth": 2, "function": "test2"}, {"depth": 3, "function": "test1"}]
#           },
#           {
#           "frames": [{"depth": 0, "function": "test1"}, {"depth": 1, "function": "test3"}, {"depth": 2, "function": "test2"}, {"depth": 3, "function": "test1"}]
#           }  
#           ]
#     }
  
#   ]
#   """

#   reports = JSON.parse(b)
#   main_frames = MutableLinkedList{Frame}()
#   st,vocab = preprocessing_stacks.preprocessStacktrace(reports, 300, filter_recursion="none")

#   @test  length(st) == 6
#   @test  st[1] == [[0x00002, 0x00003, 0x00001, 0x00003]]
#   @test  st[2] == [[0x00002, 0x00004, 0x00004, 0x00003, 0x00004, 0x00004, 0x00004, 0x00004, 0x00004, 0x00004, 0x00003]]
#   @test  st[3] == [[0x00002, 0x00003, 0x00003],[0x00005, 0x00005, 0x00003]]
#   @test  st[3] == [[0x00002, 0x00003, 0x00003],[0x00005, 0x00005, 0x00003]]

#   st,vocab = preprocessing_stacks.preprocessStacktrace(reports, 300, filter_recursion="none", unique_ukn_report=false )

#   @test  st[2] == [[0x00002, 0x00001, 0x00001, 0x00003, 0x00001, 0x00001, 0x00001, 0x00001, 0x00001, 0x00001, 0x00003]]
#   @test  st[3] == [[0x00002, 0x00003, 0x00003],[0x00001, 0x00001, 0x00003]]


#   st,vocab = preprocessing_stacks.preprocessStacktrace(reports, 300, filter_recursion="none", rm_sub_stacks=true)


#   @test  st[6] == [[0x00002, 0x00009, 0x00007, 0x00002]]


#   st,vocab = preprocessing_stacks.preprocessStacktrace(reports, 300, filter_recursion="brodie", rm_sub_stacks=true)

#   @test  st[3] == [[0x00002, 0x00003],[0x00005, 0x00005, 0x00003]]
#   @test  st[2] == [[0x00002, 0x00004, 0x00004, 0x00003, 0x00004, 0x00004, 0x00004, 0x00004, 0x00004, 0x00004, 0x00003]]


#   st,vocab = preprocessing_stacks.preprocessStacktrace(reports, 300, filter_recursion="modani", rm_sub_stacks=true)

#   @test  st[1] == [[0x00002, 0x00003]]
#   @test  st[6] == [[0x00002]]


#   @test preprocessing_stacks.removeRecursiveCalls([0x00002, 0x00003, 0x00003], 0x000000, "none") == [0x00002, 0x00003, 0x00003]

#   @test preprocessing_stacks.removeRecursiveCalls([0x00003, 0x00002, 0x00003, 0x00003, 0x000001, 0x000001,], 0x000001, "brodie") == [0x00003, 0x00002, 0x00003, 0x000001, 0x000001]

#   @test preprocessing_stacks.removeRecursiveCalls([0x00002, 0x000001, 0x00003, 0x00002, 0x00002, 0x000001, 0x000001, 0x00003, 0x00003], 0x000001, "modani") == [0x00002, 0x000001, 0x000001, 0x00003]


#   doc = DocFreq(zeros(UInt32, length(vocab)),1)


#   retrieveDocfreq!(doc, values(st))
#   @test doc.nDocs == 7 && doc.doc_freq == [0x000000, 0x000006, 0x00003, 0x00001, 0x00001, 0x000001, 0x000002, 0x00001, 0x00001, 0x00000]

#   doc = DocFreq(zeros(UInt32, length(vocab)),1)
  
#   retrieveDocfreq!(doc, values(st), true)
#   @test doc.nDocs == 10 && doc.doc_freq == [0x000000, 0x000008, 0x00004, 0x00001, 0x00001, 0x000001, 0x000002, 0x00001, 0x00001, 0x00000]
# end

