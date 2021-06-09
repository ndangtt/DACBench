#python train.py --outdir test --cores 1 --instance_set onemax_500 --bench_config "init_solution_ratio=0.95,reward_choice=imp_div_evals_new"
python train.py --outdir exp --cores 1 --instance_set onemax_500 --max_steps 2000000 --agent_config "entropy_coef=0.01; var_init=3" --bench_config "init_solution_ratio=0.95; reward_choice=imp_minus_evals" >exp/log.txt 2>&1
