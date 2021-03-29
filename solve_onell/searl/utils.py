from copy import deepcopy
import importlib
from dacbench.abstract_benchmark import objdict

def make_env(bench_config):
    config = objdict(deepcopy(bench_config.__dict__))
    del config.name
    del config.base_config_name
    module_name = '.'.join(bench_config.name.split('.')[:-1])
    class_name = bench_config.name.split('.')[-1]
    bench_class = getattr(importlib.import_module(module_name), class_name)
    print(bench_class)
    bench = bench_class(bench_config.base_config_name, config)
    env = bench.get_environment()
    return env
