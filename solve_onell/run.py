from solve_onell.pfrl_ppo import PfrlPPO
from dacbench.benchmarks.onell_benchmark import OneLLBenchmark
import pfrl
import logging
import argparse
import json
import os

def make_onell_env(seed, test, wrappers=[], bench_config=None):
    # use a different seed if this is a test environment
    if test:
        seed = 2 ** 32 - 1 - seed
    if bench_config:
        bench = OneLLBenchmark(config=bench_config)
    else:
        bench = OneLLBenchmark()
    bench.config.seed = seed
    env = bench.get_environment()
    for wrapper in wrappers:
        env = wrapper['function'](env, **wrapper['args'])
    return env

def read_agent_config(config):
    agent_config = None
    if config is not None:
        if os.path.isfile(config):  # agent config as json file
            with open(config, 'rt') as f:
                agent_config = json.load(f)
        else:  # agent config as string ("param1=val1,param2=val2,...")
            ls = [s.strip() for s in config.split(',')]
            agent_config = {}
            for s in ls:
                param = s.split('=')[0]
                val = s.split('=')[1]
                try:
                    val = int(val)
                except:
                    try:
                        val = float(val)
                    except:
                        if val=='True':
                            val = True
                        elif val=='False':
                            val = False
                        pass
                agent_config[param] = val
    return agent_config


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
    parser.add_argument("--load_and_eval", default=None, help='path to a pre-trained agent')    
    parser.add_argument("--load_and_plot", default=None, help='path to a directory of pre-trained models')       
    args = parser.parse_args()

    if args.load_and_eval:
        assert args.eval, "to load a pre-trained model and evaluate it, set --eval True"
    elif args.load_and_plot:
        assert args.eval and (args.cores==1), "to load a pre-trained model and plot it, set --eval True and --cores 1"

    config = read_agent_config(args.agent_config) 
    bench_config = None
    if args.instance_set:
        bench_config = {}
        bench_config['instance_set_path'] = args.instance_set

    ppo = PfrlPPO(make_onell_env,                 
                    config=config,
                    bench_config=bench_config,
                    n_parallel=args.cores,
                    max_steps=args.max_steps,
                    save_agent_interval = args.save_interval,
                    evaluate_during_train=args.eval,
                    eval_interval=args.eval_interval,
                    eval_n_episodes=args.eval_n_episodes,
                    outdir=args.outdir
                    )
    
    if args.load_and_eval:
        ppo.load_and_eval(args.load, args.eval_n_episodes)

    elif args.load_and_plot:
        ppo.load_and_plot(args.load_and_plot)

    else:
        ppo.train()
        


def test():
    env = make_onell_env(0,False)
    env.reset()
    env.step(env.action_space.sample())

run()
#test()
