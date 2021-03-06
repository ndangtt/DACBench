from pathlib import Path

from dacbench import benchmarks

import numpy as np
import argparse

from dacbench.logger import Logger
from dacbench.wrappers import PerformanceTrackingWrapper
from dacbench.runner import run_benchmark
from dacbench.agents import StaticAgent, GenericAgent, DynamicRandomAgent
from dacbench.envs.policies.optimal_sigmoid import get_optimum as optimal_sigmoid
from dacbench.envs.policies.optimal_luby import get_optimum as optimal_luby
from dacbench.envs.policies.optimal_fd import get_optimum as optimal_fd
from dacbench.envs.policies.csa_cma import csa
import itertools

modea_actions = [
    np.arange(2),
    np.arange(2),
    np.arange(2),
    np.arange(2),
    np.arange(2),
    np.arange(2),
    np.arange(2),
    np.arange(2),
    np.arange(2),
    np.arange(3),
    np.arange(3),
]
DISCRETE_ACTIONS = {
    "SigmoidBenchmark": np.arange(int(np.prod((5, 10)))),
    "LubyBenchmark": np.arange(6),
    "FastDownwardBenchmark": [0, 1],
    "CMAESBenchmark": [np.around(a, decimals=1) for a in np.linspace(0.2, 10, num=50)],
    "ModeaBenchmark": list(itertools.product(*modea_actions)),
    "SGDBenchmark": [np.around(a, decimals=1) for a in np.linspace(0, 10, num=50)],
}


def run_random(results_path, benchmark_name, num_episodes, seeds, fixed):
    bench = getattr(benchmarks, benchmark_name)()
    for s in seeds:
        if fixed > 1:
            experiment_name = f"random_fixed{fixed}_{s}"
        else:
            experiment_name = f"random_{s}"
        logger = Logger(
            experiment_name=experiment_name, output_path=results_path / benchmark_name
        )
        env = bench.get_benchmark(seed=s)
        env = PerformanceTrackingWrapper(
            env, logger=logger.add_module(PerformanceTrackingWrapper)
        )
        agent = DynamicRandomAgent(env, fixed)

        logger.add_agent(agent)
        logger.add_benchmark(bench)
        logger.set_env(env)
        logger.set_additional_info(seed=s)

        run_benchmark(env, agent, num_episodes, logger)

        logger.close()


def run_static(results_path, benchmark_name, action, num_episodes, seeds=np.arange(10)):
    bench = getattr(benchmarks, benchmark_name)()
    for s in seeds:
        logger = Logger(
            experiment_name=f"static_{action}_{s}",
            output_path=results_path / benchmark_name,
        )
        env = bench.get_benchmark(seed=s)
        env = PerformanceTrackingWrapper(
            env, logger=logger.add_module(PerformanceTrackingWrapper)
        )
        agent = StaticAgent(env, action)

        logger.add_agent(agent)
        logger.add_benchmark(bench)
        logger.set_env(env)
        logger.set_additional_info(seed=s, action=action)

        run_benchmark(env, agent, num_episodes, logger)

        logger.close()


def run_optimal(results_path, benchmark_name, num_episodes, seeds=np.arange(10)):
    bench = getattr(benchmarks, benchmark_name)()
    if benchmark_name == "LubyBenchmark":
        policy = optimal_luby
    elif benchmark_name == "SigmoidBenchmark":
        policy = optimal_sigmoid
    elif benchmark_name == "FastDownwardBenchmark":
        policy = optimal_fd
    elif benchmark_name == "CMAESBenchmark":
        policy = csa
    else:
        print("No comparison policy found for this benchmark")
        return

    for s in seeds:
        if benchmark_name == "CMAESBenchmark":
            experiment_name = f"csa_{s}"
        else:
            experiment_name = f"optimal_{s}"
        logger = Logger(
            experiment_name=experiment_name, output_path=results_path / benchmark_name
        )

        env = bench.get_benchmark(seed=s)
        env = PerformanceTrackingWrapper(
            env, logger=logger.add_module(PerformanceTrackingWrapper)
        )
        agent = GenericAgent(env, policy)

        logger.add_agent(agent)
        logger.add_benchmark(bench)
        logger.set_env(env)
        logger.set_additional_info(seed=s)

        run_benchmark(env, agent, num_episodes, logger)

        logger.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run simple baselines for DAC benchmarks"
    )
    parser.add_argument("--outdir", type=str, default="output", help="Output directory")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        type=str,
        default=None,
        help="Benchmarks to run baselines for",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate policy on",
    )
    parser.add_argument("--random", action="store_true", help="Run random policy")
    parser.add_argument("--static", action="store_true", help="Run static policy")
    parser.add_argument(
        "--optimal",
        action="store_true",
        help="Run optimal policy. Not available for all benchmarks!",
    )
    parser.add_argument(
        "--dyna_baseline",
        action="store_true",
        help="Run dynamic baseline. Not available for all benchmarks!",
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        type=float,
        default=None,
        help="Action(s) for static policy",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        help="Seeds for evaluation",
    )
    parser.add_argument(
        "--fixed_random", type=int, default=0, help="Fixes random actions for n steps",
    )
    args = parser.parse_args()

    if args.benchmarks is None:
        benchs = benchmarks.__all__
    else:
        benchs = args.benchmarks

    args.outdir = Path(args.outdir)

    if args.random:
        for b in benchs:
            run_random(args.outdir, b, args.num_episodes, args.seeds, args.fixed_random)

    if args.static:
        for b in benchs:

            if args.actions is None:
                actions = DISCRETE_ACTIONS[b]
            else:
                actions = args.actions
                if b == "FastDownwardBenchmark":
                    actions = [int(a) for a in actions]
            for a in actions:
                run_static(args.outdir, b, a, args.num_episodes, args.seeds)

    if args.optimal or args.dyna_baseline:
        for b in benchs:
            if b not in [
                "LubyBenchmark",
                "SigmoidBenchmark",
                "FastDownwardBenchmark",
                "CMAESBenchmark",
            ]:
                print("Option not available!")
                break

            run_optimal(args.outdir, b, args.num_episodes, args.seeds)


if __name__ == "__main__":
    main()
