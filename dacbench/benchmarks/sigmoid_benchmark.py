from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import SigmoidEnv

import numpy as np
import os
import csv

ACTION_VALUES = (5, 10)

INFO = {
    "identifier": "Sigmoid",
    "name": "Sigmoid Function Approximation",
    "reward": "Multiplied Differences between Function and Action in each Dimension",
    "state_description": [
        "Remaining Budget",
        "Shift (dimension 1)",
        "Slope (dimension 1)",
        "Shift (dimension 2)",
        "Slope (dimension 2)",
        "Action 1",
        "Action 2",
    ],
}

SIGMOID_DEFAULTS = objdict(
    {
        "action_space_class": "Discrete",
        "action_space_args": [int(np.prod(ACTION_VALUES))],
        "observation_space_class": "Box",
        "observation_space_type": np.float32,
        "observation_space_args": [
            np.array([-np.inf for _ in range(1 + len(ACTION_VALUES) * 3)]),
            np.array([np.inf for _ in range(1 + len(ACTION_VALUES) * 3)]),
        ],
        "reward_range": (0, 1),
        "cutoff": 10,
        "action_values": ACTION_VALUES,
        "slope_multiplier": 2.0,
        "seed": 0,
        "instance_set_path": "../instance_sets/sigmoid/sigmoid_2D3M_train.csv",
        "benchmark_info": INFO,
    }
)


class SigmoidBenchmark(AbstractBenchmark):
    """
    Benchmark with default configuration & relevant functions for Sigmoid
    """

    def __init__(self, config_path=None):
        """
        Initialize Sigmoid Benchmark

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        super(SigmoidBenchmark, self).__init__(config_path)
        if not self.config:
            self.config = objdict(SIGMOID_DEFAULTS.copy())

        for key in SIGMOID_DEFAULTS:
            if key not in self.config:
                self.config[key] = SIGMOID_DEFAULTS[key]

    def get_environment(self):
        """
        Return Sigmoid env with current configuration

        Returns
        -------
        SigmoidEnv
            Sigmoid environment

        """
        if "instance_set" not in self.config.keys():
            self.read_instance_set()

        env = SigmoidEnv(self.config)
        for func in self.wrap_funcs:
            env = func(env)

        return env

    def set_action_values(self, values):
        """
        Adapt action values and update dependencies

        Parameters
        ----------
        values: list
            A list of possible actions per dimension
        """
        self.config.action_values = values
        self.config.action_space_args = [int(np.prod(values))]
        self.config.observation_space_args = [
            np.array([-np.inf for _ in range(1 + len(values) * 3)]),
            np.array([np.inf for _ in range(1 + len(values) * 3)]),
        ]

    def read_instance_set(self):
        """Read instance set from file"""
        path = (
            os.path.dirname(os.path.abspath(__file__))
            + "/"
            + self.config.instance_set_path
        )
        self.config["instance_set"] = {}
        with open(path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                f = []
                inst_id = None
                for i in range(len(row)):
                    if i == 0:
                        try:
                            inst_id = int(row[i])
                        except Exception:
                            continue
                    else:
                        try:
                            f.append(float(row[i]))
                        except Exception:
                            continue

                if not len(f) == 0:
                    self.config.instance_set[inst_id] = f

    def get_benchmark(self, dimension=None, seed=0):
        """
        Get Benchmark from DAC paper

        Parameters
        -------
        dimension : int
            Sigmoid dimension, was 1, 2, 3 or 5 in the paper
        seed : int
            Environment seed

        Returns
        -------
        env : SigmoidEnv
            Sigmoid environment
        """
        self.config = objdict(SIGMOID_DEFAULTS.copy())
        if dimension == 1:
            self.set_action_values([3])
            self.config.instance_set_path = (
                "../instance_sets/sigmoid/sigmoid_1D3M_train.csv"
            )
            self.config.benchmark_info["state_description"] = [
                "Remaining Budget",
                "Shift (dimension 1)",
                "Slope (dimension 1)",
                "Action",
            ]
        if dimension == 2:
            self.set_action_values([3, 3])
        if dimension == 3:
            self.set_action_values((3, 3, 3))
            self.config.instance_set_path = (
                "../instance_sets/sigmoid/sigmoid_3D3M_train.csv"
            )
            self.config.benchmark_info["state_description"] = [
                "Remaining Budget",
                "Shift (dimension 1)",
                "Slope (dimension 1)",
                "Shift (dimension 2)",
                "Slope (dimension 2)",
                "Shift (dimension 3)",
                "Slope (dimension 3)",
                "Action 1",
                "Action 2",
                "Action 3",
            ]
        if dimension == 5:
            self.set_action_values((3, 3, 3, 3, 3))
            self.config.instance_set_path = (
                "../instance_sets/sigmoid/sigmoid_5D3M_train.csv"
            )
            self.config.benchmark_info["state_description"] = [
                "Remaining Budget",
                "Shift (dimension 1)",
                "Slope (dimension 1)",
                "Shift (dimension 2)",
                "Slope (dimension 2)",
                "Shift (dimension 3)",
                "Slope (dimension 3)",
                "Shift (dimension 4)",
                "Slope (dimension 4)",
                "Shift (dimension 5)",
                "Slope (dimension 5)",
                "Action 1",
                "Action 2",
                "Action 3",
                "Action 4",
                "Action 5",
            ]
        self.config.seed = seed
        self.read_instance_set()
        env = SigmoidEnv(self.config)
        return env
