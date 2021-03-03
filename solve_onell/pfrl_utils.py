from gym.spaces.utils import flatten_space
import pfrl
import functools
import json
import os
from torch import nn
import torch
import numpy as np

def flatten_env(self, env):
        # flatten observation space to Box so that it's supported by pfrl        
        env.observation_space = flatten_space(env.observation_space)

        # convert environment to float32 (required by pfrl)
        env = pfrl.wrappers.CastObservationToFloat32(env)
        return env


def make_batch_env(make_env_func, n_envs=1, 
                    seeds=None, test_env=False, 
                    wrappers=[
                        {'function':pfrl.wrappers.CastObservationToFloat32,'args':{}}
                        ],
                    bench_config=None):
    if seeds is None:
        seeds = np.arange(n_envs)
    return pfrl.envs.MultiprocessVectorEnv(
        [functools.partial(make_env_func, seed, test_env, wrappers, bench_config) for seed in seeds]
    )

    
def make_mlp(indim, outdim, n_hidden_layers, n_hidden_nodes, activation='Tanh', orthogonal_init=True):
    assert n_hidden_layers>0
    activation = getattr(nn, activation)
    layers = [nn.Linear(indim, n_hidden_nodes)]
    layers.append(activation)
    for i in range(n_hidden_layers-1):
        layers.append(nn.Linear(n_hidden_nodes, n_hidden_nodes))
        layers.append(activation)
    layers.append(nn.Linear(n_hidden_nodes, outdim))

    net = nn.Sequential(*layers)

    for layer in net:
        if isinstance(layer, nn.Linear):                
            if orthogonal_init:
                nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)            

    return net


class BoundByTanh(nn.Module):
    """
    convert a value to a pre-defined range using Tanh
    """
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = torch.Tensor(min_val)
        self.max_val = torch.Tensor(max_val)

    def forward(self, input):
        return self.min_val + torch.tanh(input) * (self.max_val - self.min_val)/2
        
    

