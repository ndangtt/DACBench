from solve_onell.ppo.pfrl_ppo import PfrlPPO
from dacbench.benchmarks.onell_benchmark import OneLLBenchmark
import solve_onell.ppo.pfrl_utils as pfrl_utils
import pfrl
import logging
import argparse
import json
import os
import sys
import yaml
from shutil import copyfile

def run():    
    log_format = '%(asctime)-15s: %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_format)   

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default='ppo-results')
    parser.add_argument("--config", default='conf.yml')
    args = parser.parse_args()

    with open(args.config, 'rt') as f:
        conf = yaml.load(f, Loader=yaml.Loader)

    exp_dir = args.exp_dir
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    copyfile(args.config, exp_dir + '/config.yml')
    
    exp_conf = conf['exp'] 
    ppo = PfrlPPO(pfrl_utils.make_env,                 
                    config=conf['ppo'],
                    bench_config=conf['bench'],
                    n_parallel=exp_conf['n_cores'],
                    max_steps=exp_conf['max_steps'],
                    save_agent_interval = exp_conf['save_interval'],
                    evaluate_during_train=True,
                    eval_interval=exp_conf['eval_interval'],
                    eval_n_episodes=exp_conf['eval_n_episodes'],
                    outdir=exp_dir
                    )
    
    ppo.train()
        

def test():
    bench = OneLLBenchmark()
    env = bench.get_environment()

run()
#test()
