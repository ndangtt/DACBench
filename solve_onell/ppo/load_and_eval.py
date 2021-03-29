from solve_onell.pfrl_ppo import PfrlPPO
from dacbench.benchmarks.onell_benchmark import OneLLBenchmark
import solve_onell.pfrl_utils as pfrl_utils
import pfrl
import logging
import argparse
import json
import os

def run():
    log_format = '%(asctime)-15s: %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_format)   

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_path", default='output/best')
    parser.add_argument("--cores", default=1, type=int)
    parser.add_argument("--n_runs",default=5, type=int)
    parser.add_argument("--instance_set", default=None)    
    args = parser.parse_args()
    
    agent_config_file = os.path.abspath(args.agent_path + '/../config.json')
    agent_config = pfrl_utils.read_config(agent_config_file) 

    bench_config = None
    bench_config_file = os.path.abspath(args.agent_path + "/../bench_config.json")
    if os.path.isfile(bench_config_file):
        bench_config = pfrl_utils.read_config(bench_config_file)    
    
    if args.instance_set:
        bench_config = {}
        bench_config['instance_set_path'] = args.instance_set

    exp_config_file = os.path.abspath(args.agent_path + "/../exp_config.json")
    with open(exp_config_file, 'rt') as f:
        exp_config = json.load(f)

    ppo = PfrlPPO(pfrl_utils.make_onell_env,                 
                    config=agent_config,
                    bench_config=bench_config,
                    n_parallel=args.cores,
                    max_steps=100000,                    
                    evaluate_during_train=True,                    
                    eval_n_episodes=args.n_runs,                    
                    save_configs=False
                    )
        
    ppo.load_and_eval(args.agent_path, args.n_runs)

    
run()
#test()
