from solve_onell.pfrl_ppo import PfrlPPO
from dacbench.benchmarks.onell_benchmark import OneLLBenchmark
import pfrl
import logging
import argparse

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

def run():
    log_format = '%(asctime)-15s: %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_format)   

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default='output')
    parser.add_argument("--cores", default=1)
    parser.add_argument("--max_steps", default=10000)
    parser.add_argument("--save_interval", default=5000)
    parser.add_argument("--eval",default=True)
    parser.add_argument("--eval_interval",default=5000)
    parser.add_argument("--eval_n_episodes",default=5)
    parser.add_argument("--instance_set", default=None)
    args = parser.parse_args()

    config = {
        'var_init': 1
    } 
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
                    eval_n_episodes=args.eval_n_episodes
                    )
    ppo.run()

def test():
    env = make_onell_env(0,False)
    env.reset()
    env.step(env.action_space.sample())

run()
#test()
