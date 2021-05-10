from copy import deepcopy
import importlib
from dacbench.abstract_benchmark import objdict
import numpy as np

def make_env(bench_config):
    config = objdict(deepcopy(bench_config.__dict__))
    del config.name
    del config.base_config_name
    module_name = '.'.join(bench_config.name.split('.')[:-1])
    class_name = bench_config.name.split('.')[-1]
    bench_class = getattr(importlib.import_module(module_name), class_name)

    # convert list to numpy array
    for k in config.keys():
        if type(config[k]) == list:
            if type(config[k][0]) == list:
                map(np.array, config[k])
            config[k] = np.array(config[k])
 
    bench = bench_class(bench_config.base_config_name, config)
    env = bench.get_environment()
    return env
