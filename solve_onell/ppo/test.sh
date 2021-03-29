#python train.py --outdir test --cores 2 --instance_set onemax_500 --agent_config "normalize_obs=False,obs_clip_threshold=None"
#python train.py --outdir test --cores 1 --instance_set onemax_500 --bench_config "init_solution_ratio=0.95,reward_choice=imp_div_evals_new"
python train.py --outdir test --cores 1 --instance_set onemax_500 --bench_config "init_solution_ratio=0.95,reward_choice=imp_minus_evals"
#python train.py --outdir test --cores 1 --instance_set onemax_500 --bench_config "init_solution_ratio=0.95"

#python load_and_eval.py --agent_path test/best
#python load_and_plot.py --outdir test
