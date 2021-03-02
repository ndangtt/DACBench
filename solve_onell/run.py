from solve_onell.pfrl_ppo import PfrlPPO
from dacbench.benchmarks.onell_benchmark import OneLLBenchmark
import pfrl
import logging

def make_onell_env(seed, test, wrappers=[]):
    # use a different seed if this is a test environment
    if test:
        seed = 2 ** 32 - 1 - seed
    bench = OneLLBenchmark()
    bench.config.seed = seed
    env = bench.get_environment()
    for wrapper in wrappers:
        env = wrapper['function'](env, **wrapper['args'])
    return env

def run():
    log_format = '%(asctime)-15s: %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_format)   
    config = {
        'var_init': 1
    } 
    ppo = PfrlPPO(make_onell_env, 
                    config=config,
                    n_parallel=2,
                    max_steps=10000,
                    save_agent_interval = 5000,
                    evaluate_during_train=True,
                    eval_interval=5000,
                    eval_n_episodes=2
                    )
    ppo.run()

def test():
    env = make_onell_env(0,False)
    env.reset()
    env.step(env.action_space.sample())

run()
#test()