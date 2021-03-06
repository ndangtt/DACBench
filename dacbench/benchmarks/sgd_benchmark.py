import csv
import json
import os

import numpy as np
import torch.nn as nn
from gym import spaces

from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import SGDEnv

HISTORY_LENGTH = 40
INPUT_DIM = 10

INFO = {"identifier": "LR",
        "name": "Learning Rate Adaption for Neural Networks",
        "reward": "Validation Loss",
        "state_description": ["Predictive Change Variance (Discounted Average)",
                              "Predictive Change Variance (Uncertainty)",
                              "Loss Variance (Discounted Average)",
                              "Loss Variance (Uncertainty)",
                              "Current Learning Rate",
                              "Training Loss",
                              "Validation Loss"]}

SGD_DEFAULTS = objdict(
    {
        "action_space_class": "Box",
        "action_space_args": [np.array([0]), np.array([10])],
        "observation_space_class": "Dict",
        "observation_space_type": None,
        "observation_space_args": [
            {
                "predictiveChangeVarDiscountedAverage": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,)
                ),
                "predictiveChangeVarUncertainty": spaces.Box(
                    low=0, high=np.inf, shape=(1,)
                ),
                "lossVarDiscountedAverage": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
                "lossVarUncertainty": spaces.Box(low=0, high=np.inf, shape=(1,)),
                "currentLR": spaces.Box(low=0, high=1, shape=(1,)),
                "trainingLoss": spaces.Box(low=0, high=np.inf, shape=(1,)),
                "validationLoss": spaces.Box(low=0, high=np.inf, shape=(1,)),
            }
        ],
        "reward_range": (-(10 ** 9), 0),
        "cutoff": 5e1,
        "lr": 1e-3,
        "training_batch_size": 64,
        "validation_batch_size": 64,
        "no_cuda": True,
        "beta1": 0.9,
        "beta2": 0.999,
        "seed": 0,
        "instance_set_path": "../instance_sets/sgd/sgd_train.csv",
        "benchmark_info": INFO
    }
)


class SGDBenchmark(AbstractBenchmark):
    """
    Benchmark with default configuration & relevant functions for SGD
    """

    def __init__(self, config_path=None):
        """
        Initialize SGD Benchmark

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        super(SGDBenchmark, self).__init__(config_path)
        if not self.config:
            self.config = objdict(SGD_DEFAULTS.copy())

        for key in SGD_DEFAULTS:
            if key not in self.config:
                self.config[key] = SGD_DEFAULTS[key]

    def get_environment(self):
        """
        Return SGDEnv env with current configuration

        Returns
        -------
        SGDEnv
            SGD environment
        """
        if "instance_set" not in self.config.keys():
            self.read_instance_set()

        env = SGDEnv(self.config)
        for func in self.wrap_funcs:
            env = func(env)

        return env

    def _architecture_constructor(self, arch_str):
        layer_specs = []
        layer_strs = arch_str.split('-')
        for layer_str in layer_strs:
            idx = layer_str.find('(')
            if idx == -1:
                nn_module_name = layer_str
                vargs = []
            else:
                nn_module_name = layer_str[:idx]
                vargs_json_str = '{"tmp": [' + layer_str[idx + 1:-1] + ']}'
                vargs = json.loads(vargs_json_str)["tmp"]
            layer_specs.append((getattr(nn, nn_module_name), vargs))

        def model_constructor():
            layers = [cls(*vargs) for cls, vargs in layer_specs]
            return nn.Sequential(*layers)

        return model_constructor

    def read_instance_set(self):
        """
        Read path of instances from config into list
        """

        path = (
                os.path.dirname(os.path.abspath(__file__))
                + "/"
                + self.config.instance_set_path
        )
        self.config["instance_set"] = {}
        with open(path, "r") as fh:
            reader = csv.DictReader(fh, delimiter=";")
            for row in reader:
                instance = [
                    row["dataset"],
                    int(row["seed"]),
                    self._architecture_constructor(row["architecture"]),
                ]
                self.config["instance_set"][int(row["ID"])] = instance

    def get_benchmark(self, seed=0):
        """
        Get benchmark from the LTO paper

        Parameters
        -------
        seed : int
            Environment seed

        Returns
        -------
        env : SGDEnv
            SGD environment
        """
        self.config = objdict(SGD_DEFAULTS.copy())
        self.config.seed = seed
        self.read_instance_set()
        return SGDEnv(self.config)
