from hyperopt import hp

fixed_values= {
     "filter_func": "threshold_trim",
     "upd_doc_freq": True,
}


space = {
    "alpha": hp.uniform("alpha", 0.0, 30.0),
    "beta": hp.uniform("beta", 0.0, 100.0),
    "gamma": hp.uniform("gamma", 0.0, 30.0),

    "freq_by_stacks": hp.choice("freq_by_stacks", (False, True)),
    "filter_func_k": hp.uniform("filter_func_k", 0.0, 130.0),
    "filter_recursion": hp.choice("filter_recursion", ("none", 'modani', 'brodie')),

    "aggregate": hp.choice("aggregate", ('max', 'avg_query', 'avg_cand', 'avg_short', 'avg_long', 'avg_query_cand')),

}

