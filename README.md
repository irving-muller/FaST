# FaST: A linear time stack trace alignment heuristic for crash report deduplication
By Irving Muller Rodrigues, Daniel Aloise, and Eraldo Rezende Fernandes

[Preprint](https://irving-muller.github.io/papers/FaST.pdf)

## Abstract

In software projects, applications are often monitored by systems that automatically identify crashes, collect their information into reports, and submit them to developers. Especially in popular applications, such systems tend to generate a large number of crash reports in which a significant portion of them are duplicate. Due to this high submission volume, in practice, the crash report deduplication is supported by devising automatic systems whose efficiency is a critical constraint. In this paper, we focus on improving deduplication system throughput by speeding up the stack trace comparison. In contrast to the state-of-the-art techniques, we propose FaST, a novel sequence alignment method that computes the similarity score between two stack traces in linear time. Our method independently aligns identical frames in two stack traces by means of a simple alignment heuristic. We evaluate FaST and five competing methods on four datasets from open-source projects using ranking and binary metrics. Despite its simplicity, FaST consistently achieves state-of-the-art performance regarding all metrics considered. Moreover, our experiments confirm that FaST is substantially more efficient than methods based on optimal sequence alignment.


## Setup
1. Install julia 1.6.2
2. Insert in bashrc: export JULIA_LOAD_PATH="path/FaST_code/modules:$JULIA_LOAD_PATH"
3. Run this script to install julia packages:
```
using Pkg
Pkg.add("ArgParse")
Pkg.add("JSON")
Pkg.add("Statistics")
Pkg.add("DataStructures")
Pkg.add("Dates")
Pkg.add("SHA")
```
4. Download the dataset from [link](https://zenodo.org/record/5746044/#.Yej5HvtyZH6).

## Run experiments

```
cd path/FaST

###############
## FaST
#############

#Campbell
#julia --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/campbell_dataset/campbell_stacktraces.json ${DT_PATH}/campbell_dataset/chunks_test/validation_chunk_${CHUNK_ID}.txt "plain_fast" space_script/fast_space_campbell.py --filter_func "threshold_trim" --max_evals 100  --test "${DT_PATH}/campbell_dataset/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730

# Eclipse
#julia  --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/eclipse_2018/eclipse_stacktraces.json ${DT_PATH}/eclipse_2018/chunks_test/validation_chunk_${CHUNK_ID}.txt "plain_fast" space_script/fast_space_eclipse.py --filter_func "threshold_trim" --max_evals 100  --test "${DT_PATH}/eclipse_2018/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730

#Netbeans
#julia  --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/netbeans_2016/netbeans_stacktraces.json ${DT_PATH}/netbeans_2016/chunks_test/validation_chunk_${CHUNK_ID}.txt "plain_fast" space_script/fast_space_netbeans.py --filter_func "threshold_trim" --max_evals 100  --test "${DT_PATH}/netbeans_2016/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730


#Gnome
julia  --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/gnome_2011/gnome_stacktraces.json ${DT_PATH}/gnome_2011/chunks_test/validation_chunk_${CHUNK_ID}.txt "plain_fast" space_script/fast_space_gnome.py --filter_func "threshold_trim" --max_evals 100  --test "${DT_PATH}/gnome_2011/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730


##########
# DURFEX
##############

# Eclipse
julia  --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/eclipse_2018/eclipse_stacktraces.json ${DT_PATH}/eclipse_2018/chunks_test/validation_chunk_${CHUNK_ID}.txt "plain_cosine_tf_idf" space_script/durfex_space_eclipse.py --filter_func "threshold_trim" --max_evals 30  --test "${DT_PATH}/eclipse_2018/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730


# Netbeans
#julia  --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/netbeans_2016/netbeans_stacktraces.json ${DT_PATH}/netbeans_2016/chunks_test/validation_chunk_${CHUNK_ID}.txt "plain_cosine_tf_idf" space_script/durfex_space_netbeans.py --filter_func "threshold_trim" --max_evals 30  --test "${DT_PATH}/netbeans_2016/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730



##########
# TF-IDF
##############

#Campbell
#julia  --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/campbell_dataset/campbell_stacktraces.json ${DT_PATH}/campbell_dataset/chunks_test/validation_chunk_${CHUNK_ID}.txt "plain_tf_idf" space_script/tf_idf_space_campbell.py --filter_func "threshold_trim" --max_evals 50  --test "${DT_PATH}/campbell_dataset/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730

# Eclipse
#julia  --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/eclipse_2018/eclipse_stacktraces.json ${DT_PATH}/eclipse_2018/chunks_test/validation_chunk_${CHUNK_ID}.txt "plain_tf_idf" space_script/tf_idf_space_eclipse.py --filter_func "threshold_trim" --max_evals 50  --test "${DT_PATH}/eclipse_2018/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730

# Netbeans
#julia  --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/netbeans_2016/netbeans_stacktraces.json ${DT_PATH}/netbeans_2016/chunks_test/validation_chunk_${CHUNK_ID}.txt "plain_tf_idf" space_script/tf_idf_space_netbeans.py --filter_func "threshold_trim" --max_evals 50  --test "${DT_PATH}/netbeans_2016/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730

# TF-IDF
julia  --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/gnome_2011/gnome_stacktraces.json ${DT_PATH}/gnome_2011/chunks_test/validation_chunk_${CHUNK_ID}.txt "plain_tf_idf" space_script/tf_idf_space_gnome.py --filter_func "threshold_trim" --max_evals 50  --test "${DT_PATH}/gnome_2011/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730


###############
## TraceSim
#############

#Campbell
#julia --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/campbell_dataset/campbell_stacktraces.json ${DT_PATH}/campbell_dataset/chunks_test/validation_chunk_${CHUNK_ID}.txt "trace_sim" space_script/trace_sim_space_campbell.py --filter_func "threshold_trim" --max_evals 100  --test "${DT_PATH}/campbell_dataset/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730

# Eclipse
#julia  --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/eclipse_2018/eclipse_stacktraces.json ${DT_PATH}/eclipse_2018/chunks_test/validation_chunk_${CHUNK_ID}.txt "trace_sim" space_script/trace_sim_space_eclipse.py --filter_func "threshold_trim" --max_evals 100  --test "${DT_PATH}/eclipse_2018/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730

#Netbeans
#julia  --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/netbeans_2016/netbeans_stacktraces.json ${DT_PATH}/netbeans_2016/chunks_test/validation_chunk_${CHUNK_ID}.txt "trace_sim" space_script/trace_sim_space_netbeans.py --filter_func "threshold_trim" --max_evals 100  --test "${DT_PATH}/netbeans_2016/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730

#Gnome
julia  --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/gnome_2011/gnome_stacktraces.json ${DT_PATH}/gnome_2011/chunks_test/validation_chunk_${CHUNK_ID}.txt "trace_sim" space_script/trace_sim_space_gnome.py --filter_func "threshold_trim" --max_evals 100  --test "${DT_PATH}/gnome_2011/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730


###############
## PDM
#############

#Eclipse
#julia  --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/eclipse_2018/eclipse_stacktraces.json ${DT_PATH}/eclipse_2018/chunks_test/validation_chunk_${CHUNK_ID}.txt "pdm" space_script/pdm_space_eclipse.py --filter_func "threshold_trim" --max_evals 100  --test "${DT_PATH}/eclipse_2018/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730



#Campbell
#julia --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/campbell_dataset/campbell_stacktraces.json ${DT_PATH}/campbell_dataset/chunks_test/validation_chunk_${CHUNK_ID}.txt "pdm" space_script/pdm_space_campbell.py --filter_func "threshold_trim" --max_evals 100  --test "${DT_PATH}/campbell_dataset/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730


#Netbeans
julia  --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/netbeans_2016/netbeans_stacktraces.json ${DT_PATH}/netbeans_2016/chunks_test/validation_chunk_${CHUNK_ID}.txt "pdm" space_script/pdm_space_netbeans.py --filter_func "threshold_trim" --max_evals 100  --test "${DT_PATH}/netbeans_2016/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730

# Gnome
#julia  --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/gnome_2011/gnome_stacktraces.json ${DT_PATH}/gnome_2011/chunks_test/validation_chunk_${CHUNK_ID}.txt "pdm" space_script/pdm_space_gnome.py --filter_func "threshold_trim" --max_evals 100  --test "${DT_PATH}/gnome_2011/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730


###############
## Prefix match
#############


#Eclipse
#julia  --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/eclipse_2018/eclipse_stacktraces.json ${DT_PATH}/eclipse_2018/chunks_test/validation_chunk_${CHUNK_ID}.txt "prefix_match" space_script/prefix_match_space_eclipse.py --filter_func "threshold_trim" --max_evals 100  --test "${DT_PATH}/eclipse_2018/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730



#Campbell
#julia --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/campbell_dataset/campbell_stacktraces.json ${DT_PATH}/campbell_dataset/chunks_test/validation_chunk_${CHUNK_ID}.txt "prefix_match" space_script/prefix_match_space_campbell.py --filter_func "threshold_trim" --max_evals 100  --test "${DT_PATH}/campbell_dataset/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730


#Netbeans
julia  --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/netbeans_2016/netbeans_stacktraces.json ${DT_PATH}/netbeans_2016/chunks_test/validation_chunk_${CHUNK_ID}.txt "prefix_match" space_script/prefix_match_space_netbeans.py --filter_func "threshold_trim" --max_evals 100  --test "${DT_PATH}/netbeans_2016/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730

# Gnome
#julia  --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/gnome_2011/gnome_stacktraces.json ${DT_PATH}/gnome_2011/chunks_test/validation_chunk_${CHUNK_ID}.txt "prefix_match" space_script/prefix_match_space_gnome.py --filter_func "threshold_trim" --max_evals 100  --test "${DT_PATH}/gnome_2011/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730

###############
## TSM
#############


#Campbell
#julia  --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/campbell_dataset/campbell_stacktraces.json ${DT_PATH}/campbell_dataset/chunks_test/validation_chunk_${CHUNK_ID}.txt "trace_sim" space_script/trace_sim_sum_space_campbell.py --filter_func "threshold_trim" --max_evals 100  --test "${DT_PATH}/campbell_dataset/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730

# Eclipse
#julia  --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/eclipse_2018/eclipse_stacktraces.json ${DT_PATH}/eclipse_2018/chunks_test/validation_chunk_${CHUNK_ID}.txt "trace_sim" space_script/trace_sim_sum_space_eclipse.py --filter_func "threshold_trim" --max_evals 100  --test "${DT_PATH}/eclipse_2018/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730

#Netbeans
#julia  --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/netbeans_2016/netbeans_stacktraces.json ${DT_PATH}/netbeans_2016/chunks_test/validation_chunk_${CHUNK_ID}.txt "trace_sim" space_script/trace_sim_sum_space_netbeans.py --filter_func "threshold_trim" --max_evals 100  --test "${DT_PATH}/netbeans_2016/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730


#Gnome
julia  --threads $NTHREADS experiments/hyperparameter_opt.jl ${DT_PATH}/gnome_2011/gnome_stacktraces.json ${DT_PATH}/gnome_2011/chunks_test/validation_chunk_${CHUNK_ID}.txt "trace_sim" space_script/trace_sim_sum_space_gnome.py --filter_func "threshold_trim" --max_evals 100  --test "${DT_PATH}/gnome_2011/chunks_test/test_chunk_${CHUNK_ID}.txt" -w 730
```

## Results

The experimental results can be found on: experiment_results.ipynb

## Citation
The paper was accepted and will be published in MSR 2022.

If the code helps your research, please consider to cite our work:
```
@preamble{ " \newcommand{\noop}[1]{} " }

@inproceedings{irving2022b,
  author = {Irving Muller Rodrigues and 
              Daniel Aloise and
              Eraldo Rezende Fernandes},
  title = {FaST: A linear time stack trace alignment heuristic for crash report deduplication},
  year={\noop{3001}in press},
  publisher = {},
  address = {},
  booktitle = {Proceedings of the 19th International Conference on Mining Software Repositories},
  pages = {},
  numpages = {}
}
```

