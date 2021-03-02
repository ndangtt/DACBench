"""
This file follows examples in: 
    
    
List of PPO hyper-parameters:
    https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
    https://docs.google.com/spreadsheets/d/1fNVfqgAifDWnTq-4izPPW_CVAUu9FXl3dWkqWIXz04o/htmlview

Papers discussing important implementation choices:
    https://arxiv.org/pdf/2010.13083.pdf (weight init, normalization, adaptive learning techniques for PPO, TRPO, TD3, SAC)

TODO:

    - Check if chainerrl PPO uses KL early stopping for policy updates to prevent the new policy to be too far from the old one (used in OpenAI baselines: https://spinningup.openai.com/en/latest/algorithms/ppo.html, see 1st "You should know" box)

    - Implementation of shared network needs to be revised
    - Implementation of creating policies for discrete-action needs to be revised

    - Add reward scaling using running std: 
        + Shown to be important in https://arxiv.org/pdf/2005.12729.pdf
        + Current chainerrl only support reward scaling via a fixed scaling factor (reward_scale_factor)    

    - Add extra stats for diagnostic from Schulman - Nuts and Bolts
        + KL between old and new policy
        + Baseline explained variance: (1 - var[empirical returns - predicted value]) / var[empirical returns]

    - Add policy initialisation (Schulman - Nuts and Bolts): zero or tiny final layer to maximise entropy 

    - Add support for A2C setting
    
"""

from collections import OrderedDict
from dacbench.abstract_benchmark import objdict
import logging
import solve_onell.pfrl_utils as utils
import os
import json
import pfrl
import gym
import torch
from torch import nn as nn
import numpy as np
from dacbench.benchmarks.onell_benchmark import OneLLBenchmark

PPO_DEFAULT_CONFIG = OrderedDict({        
    "normalize_obs": True, # standardise observations based on their running empirical mean and variance 
    "obs_clip_threshold": None, # (conditional: normalize_obs=True) clip all normalized obs to [-threshold, threshold]. Set to None for no threshold
    "reward_scale_factor": 1.0, # reward = reward * reward_scale_factor           

    # policy and value function networks
    "n_hidden_nodes": 50, # number of hidden nodes    
    "activation": "Tanh", # activation function
    "bound_mean": False, # (continuous action space only) bound the learnt mean action to [min_action, max_action] using tanh
    "var_init": 0, # (continuous action space only) initial value for variance of gausian distribution (assuming state-indenpendent variance), must be >=0, for policy network only    
    "orthogonal_init": True, # orthogonal initialization for network weights instead of Xavier, shown to be important in https://arxiv.org/pdf/2005.12729.pdf
    "zero_init_bias": True, # set all initial bias weights as zero (used in pfrl mujoco examples for PPO and TRPO)

    "lr": 3e-4, # learning rate 
    "lr_linear_decay": True, # linearly decay lr to 0 (shown to be important in https://arxiv.org/pdf/2005.12729.pdf, param: anneal-lr) 
       
    "update_interval": 2048, # number of steps per policy iteration (~amount of data used for each model update iteration, some sources say that this value should be large, i.e., more data per policy update iteration even though it means less iterations)
    "batchsize": 64, # minibatch size    
    "n_epochs": 10, # number of epochs per policy update

    "vf_coef": 1.0, # weight coefficient for loss of value function
                    # TODO: does this parameter matter if shared_network=False?
    "entropy_coef": 0.01, # weight coefficient for entropy bonus

    "clip_eps": 0.2, # clip parameter (epsilon) for policy function
    "clip_eps_vf": None, # clip parameter (epsilon) for value function (doesn't seem to be important in https://arxiv.org/pdf/2005.12729.pdf), and should be used with care when in combination with reward scaling (https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/ppo2/ppo2.html)
    "clip_eps_linear_decay": False, # linearly decay clip parameter (epsilon) for policy function    
    
    "standadize_advantages": True, # standadise all advantage values before backprop, this normally helps to boost empirical performance (see http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/hw2_final.pdf, and http://karpathy.github.io/2016/05/31/rl/)    

    "gamma": 0.99, # discount factor
    "lambd": 0.95, # GAE lambda 

    # statistics
    "value_stats_window": 1000, # window size used to compute statistics of value predictions
    "entropy_stats_window": 1000, # window size used to compute statistics of entropy of action distributions
    "value_loss_stats_window": 256, # window size used to compute statistics of loss values regarding the value function.
    "policy_loss_stats_window": 256 # window size used to compute statistics of loss values regarding the policy.
})


class PfrlPPO():
    def __init__(self, make_env_func, max_steps,
                    config=None, logger=None, n_parallel=1,
                    seed = 0,
                    save_agent_interval=1000,                                           
                    evaluate_during_train=False, 
                    eval_interval=1000, 
                    eval_n_episodes=5,                                        
                    outdir='./output'):                            
        assert max_steps>0        

        # set logger
        self.logger = logger or logging.getLogger(__name__)        

        # load config
        self.config = objdict(PPO_DEFAULT_CONFIG)
        if config:
            for key, val in config.items():
                self.config[key] = val
        config = self.config

        # create output dir
        os.makedirs(outdir, exist_ok=True)

        # save config to output dir
        with open(outdir + '/config.json', 'wt') as f:
            json.dump(self.config, f, indent=2)   

        # make sure env observation space is converted to float32 (required by pfrl)
        float32_wrapper = {'function':pfrl.wrappers.CastObservationToFloat32,'args':{}}
        self.train_env_wrappers = [float32_wrapper]
        self.eval_env_wrappers = [float32_wrapper]
        
        # reward scaling        
        if "reward_scale_factor" in config:
            self.logger.info("Scale reward during training with a factor of " + str(config.reward_scale_factor))
            self.train_env_wrappers.append({'function': pfrl.wrappers.ScaleReward,
                                    'args': {'scale':config.reward_scale_factor}})            

        # set seed for pfrl
        pfrl.utils.set_random_seed(seed)

        # set seed for environments        
        env_seeds = np.arange(n_parallel) + seed * n_parallel
        assert env_seeds.max() < 2 ** 32

        # create a batch of training and evaluation envs        
        self.train_envs = utils.make_batch_env(make_env_func, n_envs=n_parallel, seeds=env_seeds,test_env=False, wrappers=self.train_env_wrappers)
        if evaluate_during_train:
            self.eval_envs = utils.make_batch_env(make_env_func, n_envs=n_parallel, seeds=env_seeds,test_env=True, wrappers=self.eval_env_wrappers)

        
        # get info about env's state space and action space
        # (used for setting up the agent)
        env = make_env_func(test=False, seed=0)
        obs_space = env.observation_space
        obs_size = obs_space.low.size
        act_space = env.action_space                

        # observation normalizer
        obs_normalizer = None
        if config.normalize_obs:
            self.logger.info("Normalize observation space")
            threshold = config.obs_clip_threshold
            if threshold is not None:
                self.logger.info("Clip observation space within [-" + str(threshold) + ", " + str(threshold) + "]")
            obs_normalizer = pfrl.nn.EmpiricalNormalization(obs_space.low.size, clip_threshold=threshold)   

        # info to create networks
        activation = getattr(nn, config.activation) 
        n_hiddens = config.n_hidden_nodes   
        act_size = act_space.low.size

        # create policy network
        policy_layers = [
            nn.Linear(obs_size, n_hiddens),
            activation(),
            nn.Linear(n_hiddens, n_hiddens),
            activation(),
            nn.Linear(n_hiddens, act_size),                
            
        ]
        if isinstance(act_space, gym.spaces.Box):
            if config.bound_mean:                
                policy_layers.append(utils.BoundByTanh(act_space.low, act_space.high))
            policy_layers.append(pfrl.policies.GaussianHeadWithStateIndependentCovariance(
                action_size=act_size,
                var_type="diagonal",
                var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
                var_param_init=config.var_init,  # log std = 0 => std = 1
                ))
        else:
            policy_layers.append(pfrl.policies.SoftmaxCategoricalHead())        
        policy = nn.Sequential(*policy_layers)

        # create value function network   
        vf = torch.nn.Sequential(
            nn.Linear(obs_size, n_hiddens),
            activation(),
            nn.Linear(n_hiddens, n_hiddens),
            activation(),
            nn.Linear(n_hiddens, 1)
        )

        # orthogonal weight initialisation
        if config.orthogonal_init:
            for i in [0,2,4]:
                nn.init.orthogonal_(policy[i].weight, gain=1.0)
                nn.init.orthogonal_(vf[i].weight, gain=1.0)

        # bias weight zero initialisation
        if config.zero_init_bias:
            for i in range(0,5,2):
                nn.init.zeros_(policy[i].bias)
                nn.init.zeros_(vf[i].bias) 

        # combine policy and vf into one model, since we're using one optimiser for both networks
        model = pfrl.nn.Branched(policy, vf)           
        
        self.logger.debug("PPO agent model: ")
        self.logger.debug(model)

        # create optimizer
        opt = torch.optim.Adam(model.parameters(), lr=config.lr)

        # add hook for linearly decaying learning rate to zero    
        self.step_hooks = []    
        if config.lr_linear_decay:
            def lr_setter(env, agent, value):
                for param_group in agent.optimizer.param_groups:
                    param_group["lr"] = value
            lr_decay_hook = pfrl.experiments.LinearInterpolationHook(max_steps, config.lr, 0, lr_setter)
            self.step_hooks.append(lr_decay_hook)
        
        # add hook for lineary decaying epsilon (clip param for policy)
        if config.clip_eps_linear_decay:
            def clip_eps_setter(env, agent, value):
                agent.clip_eps = max(value, 1e-8)
            clip_eps_decay_hook = pfrl.experiments.LinearInterpolationHook(max_steps, config.clip_eps, 0, clip_eps_setter) 
            self.step_hooks.append(clip_eps_decay_hook)

        # clip_eps_vf is not yet supported
        if config.clip_eps_vf is not None:
            print("Value function clipping for PPO is not yet supported, as I don't know how it will affect reward scaling.")
            raise NotImplementedError

        # create PPO agent
        self.agent = pfrl.agents.PPO(model, opt, obs_normalizer=obs_normalizer,
                        gamma=config.gamma, lambd=config.lambd, 
                        value_func_coef=config.vf_coef, 
                        entropy_coef=config.entropy_coef,
                        update_interval=config.update_interval, 
                        minibatch_size=config.batchsize, epochs=config.n_epochs,
                        clip_eps=config.clip_eps, clip_eps_vf=config.clip_eps_vf,
                        standardize_advantages=config.standadize_advantages,
                        value_stats_window=config.value_stats_window,
                        entropy_stats_window=config.entropy_stats_window,
                        value_loss_stats_window=config.value_loss_stats_window,
                        policy_loss_stats_window=config.policy_loss_stats_window)

        # save experiment config
        self.exp_config = objdict()
        for name in ['max_steps','n_parallel','seed','save_agent_interval','evaluate_during_train','eval_interval','eval_n_episodes','outdir']:
            self.exp_config[name] = locals()[name]        
        with open(outdir + '/exp_config.json', 'wt') as f:
            json.dump(self.exp_config, f, indent=2) 

    def run(self):
        config = self.exp_config
        if self.exp_config.evaluate_during_train:
            pfrl.experiments.train_agent_batch_with_evaluation(
                agent=self.agent,
                env=self.train_envs, 
                steps=config.max_steps,
                checkpoint_freq=config.save_agent_interval,    
                eval_env=self.eval_envs, 
                eval_n_steps=None,
                eval_n_episodes=config.eval_n_episodes,
                eval_interval=config.eval_interval,
                outdir=config.outdir,
                step_hooks=self.step_hooks,
                logger=self.logger                
            )
        else:
            pfrl.experiments.train_agent_batch(
                agent=self.agent,
                env=self.train_envs, 
                steps=config.max_steps,
                checkpoint_freq=config.save_agent_interval,                    
                outdir=config.outdir,
                step_hooks=self.step_hooks,
                logger=self.logger            
            )
