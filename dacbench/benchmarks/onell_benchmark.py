from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs.onell_env import OneLLEnv, RLSEnv

import numpy as np
import os
import pandas as pd

class OneLLBenchmark(AbstractBenchmark):
    """
    Benchmark with various settings for (1+(lbd, lbd))-GA
    """

    def __init__(self, base_config_name=None, config=None):
        """
        Initialize OneLL benchmark

        Parameters            
        -------
        base_config_name: str
            OneLL's config name
            possible values: see ../additional_configs/onell/configs.py
        config : str
            a dictionary, all options specified in this argument will override the one in base_config_name

        """
        if base_config_name is None:   
            base_config_name = 'lbd_theory'
        config_path = os.path.dirname(os.path.abspath(__file__)) + '/../additional_configs/onell/' + base_config_name + '.json'        
        super(OneLLBenchmark, self).__init__(config_path)        

        if config:
            for key, val in config.items():
                self.config[key] = val        

        self.read_instance_set()                   


    def get_environment(self):
        """
        Return an environment with current configuration        
        """        
        assert self.config.algorithm in ['onell','rls']
        
        if self.config.algorithm == "onell":
            env = OneLLEnv(self.config)
        else:
            env = RLSEnv(self.config)
        for func in self.wrap_funcs:
            env = func(env)

        return env

    
    def read_instance_set(self):
        """Read instance set from file"""        
        assert self.config.instance_set_path        
        if os.path.isfile(self.config.instance_set_path):
            path = self.config.instance_set_path
        else:        
            path = (
                os.path.dirname(os.path.abspath(__file__))
                + "/../instance_sets/onell/"
                + self.config.instance_set_path + ".csv"
            )                

        self.config["instance_set"] = pd.read_csv(path,index_col=0).to_dict('id')

        for key, val in self.config['instance_set'].items():
            self.config['instance_set'][key] = objdict(val)
