from solve_onell.ppo.pfrl_ppo import PfrlPPO
from dacbench.benchmarks.onell_benchmark import OneLLBenchmark
import solve_onell.ppo.pfrl_utils as pfrl_utils
import pfrl
import logging
import argparse
import json
import os

def run():    
    log_format = '%(asctime)-15s: %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_format)   

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default='output')
    parser.add_argument("--cores", default=1, type=int)
    parser.add_argument("--max_steps", default=10000, type=int)
    parser.add_argument("--save_interval", default=5000, type=int)
    parser.add_argument("--eval",default=True, type=int)
    parser.add_argument("--eval_interval",default=5000, type=int)
    parser.add_argument("--eval_n_episodes",default=5, type=int)
    parser.add_argument("--instance_set", default=None)
    parser.add_argument("--agent_config",default=None)
    parser.add_argument("--bench_config",default=None)
    args = parser.parse_args()
    
    agent_config = pfrl_utils.read_config(args.agent_config) 
    bench_config = pfrl_utils.read_config(args.bench_config)
        
    if args.instance_set:
        if bench_config is None:
            bench_config = {}
        bench_config['instance_set_path'] = args.instance_set    

    ppo = PfrlPPO(pfrl_utils.make_onell_env,                 
                    config=agent_config,
                    bench_config=bench_config,
                    n_parallel=args.cores,
                    max_steps=args.max_steps,
                    save_agent_interval = args.save_interval,
                    evaluate_during_train=args.eval,
                    eval_interval=args.eval_interval,
                    eval_n_episodes=args.eval_n_episodes,
                    outdir=args.outdir
                    )
    
    ppo.train()
        

def test():
    bench = OneLLBenchmark()
    env = bench.get_environment()

run()
#test()
