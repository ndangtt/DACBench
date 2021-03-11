#python train.py --outdir test --cores 2 --instance_set onemax_500 --agent_config "normalize_obs=False,obs_clip_threshold=None"
#python load_and_eval.py --agent_path test/best
python load_and_plot.py --outdir test
