
using Test
using DataStructures

using Method
using similarity_scorer
using util


# @testset "TraceSim" begin
    # CreateFrame

# Basic args
query = Report(1,2, 99999,9, [[5, 0, 2, 1, 0, 3, 5].+1])
candidates = [
    Report(2, 2, 99999,1, [[2,2,2,3,3].+1]  ),
    Report(3, 3, 99999,1, [[5, 0, 5, 5].+1]),
    Report(4, 4, 99999,2, [[2, 0, 3, 5].+1]),
    Report(5, 5, 99999,2, [[2, 0, 3].+1]),
    Report(6, 6, 99999,2, [[2, 3, 0, 0, 0, 0].+1]),
    Report(7, 7, 99999,3, [[2, 3, 5].+1]),
    Report(22, 7, 99999,3,[[0,5].+1]),
    Report(8, 8, 99999,3, [[5, 0, 2,1, 0, 3, 5].+1]),
    Report(9, 8, 99999,3, [[0, 2,1, 0].+1]),
    Report(10, 8, 99999,3, [[0, 1, 0].+1]),
    Report(11, 8, 99999,3, [[5, 0, 2, 5,1, 0,5, 3, 5, 5, 5].+1]),
    Report(12, 8, 99999,3, [[5,5, 5, 4,0,2,1,0,3].+1]),
    Report(13, 8, 99999,3, [[2,0,0,0,3,5,0].+1]),
    Report(14, 8, 99999,3, [[4,4,4].+1]),
    Report(15, 8, 99999,3, [[1,1,1].+1]),
    Report(16, 8, 99999,3, [[6,6,6].+1]),
    Report(20, 8, 99999,3, [[5, 0, 2,1, 0, 3, 5].+1, [1, 6, 4, 5].+1]),
]


####################################################
#TraceSim
#####################################################
update_doc_freq = false
docFreq = DocFreq([1.0,10.0,32.0,45.0,5.0, 95.0, 1.0], 100.0, false, false, Vocab(Dict{String, UInt32}(), Set{UInt32}()))
tracesim_scorer = createScorer("trace_sim", Dict(:alpha => 0.01, :beta => 0.15, :gamma => 0.05, :aggregate => "max"), update_doc_freq)

sim_score = computeSimilarity(tracesim_scorer, query, candidates, docFreq)
correct = [-0.9904919037079588, -0.11472926756818055, -0.11787366205090863, -0.11787427557957607, -0.07536772003384143, -0.9907868321494305, -0.13338861187092085, 0.9999999999999999, 0.9500711410717556, 0.9162800240568809, 0.9729923366154177, 0.499064501272655, -0.07400752021901913, -0.9999999999999999, -0.8187950252761799, -0.9999999999999999, 0.9999999999999999]
          
@test sum([ abs(p - c) for (p,c) in zip(sim_score, correct)]) < 0.0000000000001

####################################################
#TraceSim
#####################################################
update_doc_freq = false
docFreq = DocFreq([1.0,10.0,32.0,45.0,5.0, 95.0, 1.0], 100.0, false, false, Vocab(Dict{String, UInt32}(), Set{UInt32}()))
tracesim_scorer = createScorer("trace_sim", Dict(:alpha => 0.01, :beta => 0.15, :gamma => 0.05, :aggregate => "max", :sum_match => true), update_doc_freq)

sim_score = computeSimilarity(tracesim_scorer, query, candidates, docFreq)
correct = [-0.9810761264539718, 0.22729656910364532, 0.2247107963396082, 0.22471022824203452, 0.25947186373507225, -0.9817620596400948, 0.1989736303894532, 1.0, 0.9506488376183165, 0.9208506233979556, 0.9730051553273992, 0.659816610255179, 0.2608819681540414, -0.9999999999999999, -0.6687815256720939, -0.9999999999999998, 1.0]

@test sum([ abs(p - c) for (p,c) in zip(sim_score, correct)]) < 0.0000000000001


#####################################################
# FaST
######################################################



# update_doc_freq = false
docFreq = DocFreq([1.0,10.0,32.0,45.0,5.0, 95.0, 1.0], 100.0, false, false, Vocab(Dict{String, UInt32}(), Set{UInt32}()))
tracesim_scorer = createScorer("fast", Dict(:alpha => 0.01, :beta => 0.15, :gamma => 0.05), update_doc_freq)

for c in candidates
    updateIndex!(tracesim_scorer, c, docFreq)
end


sim_score = computeSimilarity(tracesim_scorer, query, candidates, docFreq)
correct = [-0.9818319454088075, 0.22729655111128516, 0.23586601777781668, 0.2358654439216235, 0.25130870036190883, -0.9817619907843192, 0.19897371805935532, 1.0, 0.9506488376183168, 0.9208506233979558, 0.9730051433106478, 0.6598171371957481, 0.2528600313277979, -1.0, -0.68240034277783, -1.0, 0.41822858264075174]
@test sum([ abs(p - c) for (p,c) in zip(sim_score, correct)]) < 0.0000000000001


# update_doc_freq = true
docFreq = DocFreq([1.0,10.0,32.0,45.0,5.0, 95.0, 1.0], 100.0, false, false, Vocab(Dict{String, UInt32}(), Set{UInt32}()))
tracesim_scorer = createScorer("fast", Dict(:alpha => 0.01, :beta => 0.15, :gamma => 0.05), true)

for c in candidates
    updateIndex!(tracesim_scorer, c, docFreq)
end


sim_score = computeSimilarity(tracesim_scorer, query, candidates, docFreq)
correct = [-0.9818319454088075, 0.22729655111128516, 0.23586601777781668, 0.2358654439216235, 0.25130870036190883, -0.9817619907843192, 0.19897371805935532, 1.0, 0.9506488376183168, 0.9208506233979558, 0.9730051433106478, 0.6598171371957481, 0.2528600313277979, -1.0, -0.68240034277783, -1.0, 0.41822858264075174]
@test sum([ abs(p - c) for (p,c) in zip(sim_score, correct)]) < 0.0000000000001

# update_doc_freq = true and without inverted index
docFreq = DocFreq([1.0,10.0,32.0,45.0,5.0, 95.0, 1.0], 100.0, false, false, Vocab(Dict{String, UInt32}(), Set{UInt32}()))
tracesim_scorer = createScorer("plain_fast", Dict(:alpha => 0.01, :beta => 0.15, :gamma => 0.05, :aggregate => "max"), true)

for c in candidates
    updateIndex!(tracesim_scorer, c, docFreq)
end


sim_score = computeSimilarity(tracesim_scorer, query, candidates, docFreq)
correct = [-0.9818319454088075, 0.22729655111128516, 0.23586601777781668, 0.2358654439216235, 0.25130870036190883, -0.9817619907843192, 0.19897371805935532, 1.0, 0.9506488376183168, 0.9208506233979558, 0.9730051433106478, 0.6598171371957481, 0.2528600313277979, -1.0, -0.68240034277783, -1.0, 1.0]
@test sum([ abs(p - c) for (p,c) in zip(sim_score, correct)]) < 0.0000000000001

#####################################################
# PDM
######################################################

docFreq = DocFreq([1.0,10.0,32.0,45.0,5.0, 95.0, 1.0], 100.0, false, false, Vocab(Dict{String, UInt32}(), Set{UInt32}()))
pdm_scorer = createScorer("pdm", Dict(:c => 0.025, :o => 0.5, :aggregate => "max"), update_doc_freq)

sim_score = computeSimilarity(pdm_scorer, query, candidates, docFreq)

correct = [0.31519293160643, 0.5662058455511825, 0.361822290327487, 0.40578914181401404, 0.2652418170794594, 0.21479574188557565, 0.34758544384275125, 1.0, 0.6065306597126334, 0.4494266855779494, 0.7533864775325178, 0.32323039135752085, 0.403253881818714, 0.0, 0.19714404763981902, 0.0, 1.0]
@test sum([ abs(p - c) for (p,c) in zip(sim_score, correct)]) < 0.0000000000001



#####################################################
# TF-IDF
######################################################
# println("TF=sqrt IDF=smoothing_plus")

docFreq = DocFreq([1.0,10.0,32.0,45.0,5.0, 95.0, 1.0], 100.0, false, false, Vocab(Dict{String, UInt32}(), Set{UInt32}()))
tf_idf_scorer = createScorer("tf_idf", Dict(:tf_scheme => "sqrt", :idf_scheme => "smooth_plus"), update_doc_freq)

for c in candidates
    updateIndex!(tf_idf_scorer, c, docFreq)
end

correct =  [2.7709391411745408, 3.3709821955076893, 4.938918868690841, 5.096308591777404, 5.613021564510629, 2.8612696460425804, 4.223368959658597, 5.884549758693076, 6.148279822612727, 5.876240261182948, 5.022242747210287, 5.301009940103894, 5.593802793093159, 0.0, 3.217225244042889, 0.0, 5.196743448256146]
sim_score = computeSimilarity(tf_idf_scorer, query, candidates, docFreq)
@test sum([ abs(p - c) for (p,c) in zip(sim_score, correct)]) < 0.0000000000001

# Plain tf-idf
docFreq = DocFreq([1.0,10.0,32.0,45.0,5.0, 95.0, 1.0], 100.0, false, false, Vocab(Dict{String, UInt32}(), Set{UInt32}()))
tf_idf_scorer = createScorer("plain_tf_idf", Dict(:tf_scheme => "sqrt", :idf_scheme => "smooth_plus", :aggregate => "max"), update_doc_freq)

for c in candidates
    updateIndex!(tf_idf_scorer, c, docFreq)
end

correct =  [2.7709391411745408, 3.3709821955076893, 4.938918868690841, 5.096308591777404, 5.613021564510629, 2.8612696460425804, 4.223368959658597, 5.884549758693076, 6.148279822612727, 5.876240261182948, 5.022242747210287, 5.301009940103894, 5.593802793093159, 0.0, 3.217225244042889, 0.0, 5.884549758693076]
sim_score = computeSimilarity(tf_idf_scorer, query, candidates, docFreq)
@test sum([ abs(p - c) for (p,c) in zip(sim_score, correct)]) < 0.0000000000001



# println("TF=raw IDF=std")
docFreq = DocFreq([1.0,10.0,32.0,45.0,5.0, 95.0, 1.0], 100.0, false, false, Vocab(Dict{String, UInt32}(), Set{UInt32}()))
tf_idf_scorer = createScorer("tf_idf", Dict(:tf_scheme => "raw", :idf_scheme => "std"), update_doc_freq)

for c in candidates
    updateIndex!(tf_idf_scorer, c, docFreq)
end

correct =  [2.2429185035816332, 2.3795250345753716, 3.2972027298908895, 3.7776675700282736, 8.311372923009866, 1.1484855208062843, 3.2926169033211283, 5.122724111094879, 6.326179874079298, 6.646990616360661, 4.1483821898428666, 4.534915775846339, 7.714223151809972, 0.0, 3.9881943698163966, 0.0, 4.796241186760375]
sim_score = computeSimilarity(tf_idf_scorer, query, candidates, docFreq)


@test sum([ abs(p - c) for (p,c) in zip(sim_score, correct)]) < 0.0000000000001




#println("TF=sqrt IDF=smooth")
docFreq = DocFreq([1.0,10.0,32.0,45.0,5.0, 95.0, 1.0], 100.0, false, false, Vocab(Dict{String, UInt32}(), Set{UInt32}()))
tf_idf_scorer = createScorer("tf_idf", Dict(:tf_scheme => "sqrt", :idf_scheme => "smooth"), update_doc_freq)

for c in candidates
    updateIndex!(tf_idf_scorer, c, docFreq)
end

correct =  [1.3638869398993816, 2.004956791723251, 2.9389188686908403, 3.364257784208526, 3.9800284026551775, 1.1292188384737032, 2.8091553972855023, 3.681611372015697, 4.441173041426179, 4.482393411065596, 2.9527583348897792, 3.252255150123237, 3.7039804280470228, 0.0, 2.217225244042889, 0.0, 3.2186849258110835]
sim_score = computeSimilarity(tf_idf_scorer, query, candidates, docFreq)
@test sum([ abs(p - c) for (p,c) in zip(sim_score, correct)]) < 0.0000000000001


#println("TF=sqrt IDF=smooth Add_query_freq")
docFreq = DocFreq([1.0,10.0,32.0,45.0,5.0, 95.0, 1.0], 100.0, false, false, Vocab(Dict{String, UInt32}(), Set{UInt32}()))
tf_idf_scorer = createScorer("tf_idf", Dict(:tf_scheme => "sqrt", :idf_scheme => "smooth", :alpha => 1.5, :beta => 0.75, :add_query_freq => true), update_doc_freq)

for c in candidates
    updateIndex!(tf_idf_scorer, c, docFreq)
end

correct =  [1.3638869398993816, 2.8354370868270706, 3.7617014351489746, 4.302183242315374, 5.306455306004881, 1.141360852825124, 3.9727456616547374, 4.561202934157668, 5.58989243599015, 5.8088203144153, 3.6609943102311915, 4.030210094183971, 4.9399619303121876, 0.0, 2.217225244042889, 0.0, 3.922371649425096]
sim_score = computeSimilarity(tf_idf_scorer, query, candidates, docFreq)


# println("TF=log IDF=smoothing_plus")
docFreq = DocFreq([1.0,10.0,32.0,45.0,5.0, 95.0, 1.0], 100.0, false, false, Vocab(Dict{String, UInt32}(), Set{UInt32}()))
tf_idf_scorer = createScorer("tf_idf", Dict(:tf_scheme => "log", :idf_scheme => "smooth_plus"), update_doc_freq)

for c in candidates
    updateIndex!(tf_idf_scorer, c, docFreq)
end

correct =  [2.1911986387721285, 2.434165845160316, 3.4233976888473707, 3.5324919316539325, 4.339031867690828, 1.9832809879761673, 2.927416286851746, 4.346041860452866, 4.55293079779292, 4.409426655093274, 3.735382769044006, 3.9336114368243877, 4.292449513590649, 0.0, 2.5749944486497767, 0.0, 3.951398702068136]
sim_score = computeSimilarity(tf_idf_scorer, query, candidates, docFreq)
@test sum([ abs(p - c) for (p,c) in zip(sim_score, correct)]) < 0.0000000000001


#####################################################
# Cosine + TF-IDF
######################################################
# Vanilla
docFreq = DocFreq([1.0,10.0,32.0,45.0,5.0, 95.0, 1.0], 100.0, false, false, Vocab(Dict{String, UInt32}(), Set{UInt32}()))
tf_idf_scorer = createScorer("plain_cosine_tf_idf", Dict(:tf_scheme => "sqrt", :idf_scheme => "smooth_plus", :aggregate => "max"), update_doc_freq)

for c in candidates
    updateIndex!(tf_idf_scorer, c, docFreq)
end

correct = [0.3327836127944191, 0.8499142010910101, 0.9135624398370668, 0.8958811351325494, 0.8990446674133254, 0.37622189797293715, 0.8587839452601135, 1.0, 0.9598906494286553, 0.9252274787322317, 0.9921356795255489, 0.9085961699138367, 0.9126623508555618, 0.0, 0.38818010385447005, 0.0, 1.0]
sim_score = computeSimilarity(tf_idf_scorer, query, candidates, docFreq)
@test sum([ abs(p - c) for (p,c) in zip(sim_score, correct)]) < 0.0000000000001


# Inverted index
docFreq = DocFreq([1.0,10.0,32.0,45.0,5.0, 95.0, 1.0], 100.0, false, false, Vocab(Dict{String, UInt32}(), Set{UInt32}()))
tf_idf_scorer = createScorer("cosine_tf_idf", Dict(:tf_scheme => "sqrt", :idf_scheme => "smooth_plus"), update_doc_freq)

for c in candidates
    updateIndex!(tf_idf_scorer, c, docFreq)
end

correct = [0.3327836127944191, 0.8499142010910101, 0.9135624398370668, 0.8958811351325494, 0.8990446674133254, 0.37622189797293715, 0.8587839452601135, 1.0, 0.9598906494286553, 0.9252274787322317, 0.9921356795255489, 0.9085961699138367, 0.9126623508555618, 0.0, 0.38818010385447005, 0.0, 0.8127041993573164]
sim_score = computeSimilarity(tf_idf_scorer, query, candidates, docFreq)
@test sum([ abs(p - c) for (p,c) in zip(sim_score, correct)]) < 0.0000000000001



#####################################################
# Durfex
######################################################

# update_doc_freq=true
ngram = 4
vocab = Vocab(Dict{String, UInt32}(), Set{UInt32}())

for i=1:7
    setdefault!(vocab, string(i))
end

dict = Dict{Int64, Report}()
dict[query.id] = query

for cand in candidates
    dict[cand.id] = cand
end

add_ngram_to_vocab!(vocab, dict, ngram)

docFreq = DocFreq(vcat([1.0,10.0,32.0,45.0,5.0, 95.0, 1.0], ones(vocabLength(vocab) - 7) .* 10.0 ), 100.0, false, false, vocab)
scorer = createScorer("cosine_tf_idf", Dict(:tf_scheme => "sqrt", :idf_scheme => "smooth_plus", :ngram =>  3), true)

n_candidates = Vector(candidates)

for c in n_candidates
    updateIndex!(scorer, c, docFreq)
end

correct = [0.09466334556581665, 0.393283530754092, 0.5993488499776823, 0.48777232617829414, 0.4010779767906943, 0.22975040384014572, 0.4438612808926345, 1.0, 0.7938605751967711, 0.5399511145828649, 0.5500360044340588, 0.6771632466375556, 0.5952515893722328, 0.0, 0.16837587375620178, 0.0, 0.8218736958280581]
sim_score = computeSimilarity(scorer, query, n_candidates, docFreq)
@test sum([ abs(p - c) for (p,c) in zip(sim_score, correct)]) < 0.0000000000001

# update_doc_freq=false 
ngram = 4
vocab = Vocab(Dict{String, UInt32}(), Set{UInt32}())

for i=1:7
    setdefault!(vocab, string(i))
end

dict = Dict{Int64, Report}()
dict[query.id] = query

for cand in candidates
    dict[cand.id] = cand
end

add_ngram_to_vocab!(vocab, dict, ngram)

docFreq = DocFreq(vcat([1.0,10.0,32.0,45.0,5.0, 95.0, 1.0], ones(vocabLength(vocab) - 7) .* 10.0 ), 100.0, false, false, vocab)
scorer = createScorer("cosine_tf_idf", Dict(:tf_scheme => "sqrt", :idf_scheme => "smooth_plus", :ngram =>  3), false)

n_candidates = Vector(candidates)

for c in n_candidates
    updateIndex!(scorer, c, docFreq)
end

correct = [0.09466334556581665, 0.393283530754092, 0.5993488499776823, 0.48777232617829414, 0.4010779767906943, 0.22975040384014572, 0.4438612808926345, 1.0, 0.7938605751967711, 0.5399511145828649, 0.5500360044340588, 0.6771632466375556, 0.5952515893722328, 0.0, 0.16837587375620178, 0.0, 0.8218736958280581]
sim_score = computeSimilarity(scorer, query, n_candidates, docFreq)
@test sum([ abs(p - c) for (p,c) in zip(sim_score, correct)]) < 0.0000000000001

#####################################################
# Prefix Match
######################################################
docFreq = DocFreq([1.0,10.0,32.0,45.0,5.0, 95.0, 1.0], 100.0, false, false, Vocab(Dict{String, UInt32}(), Set{UInt32}()))
scorer = createScorer("prefix_match", Dict(:aggregate => "max"), update_doc_freq)

n_candidates = Vector(candidates)

push!(n_candidates, Report(1,2, 99999,9, [[5, 0, 2, 1, 0, 3, 5,4,5,6].+1]))
push!(n_candidates, Report(1,2, 99999,9, [[5, 0, 2, 1, 0, 3].+1]))

for c in n_candidates
    updateIndex!(scorer, c, docFreq)
end

correct = [0.0, 0.2857142857142857, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2727272727272727, 0.1111111111111111, 0.0, 0.0, 0.0,0.0,1.0, 0.7, 0.8571428571428571]
sim_score = computeSimilarity(scorer, query, n_candidates, docFreq)

@test sum([ abs(p - c) for (p,c) in zip(sim_score, correct)]) < 0.0000000000001

