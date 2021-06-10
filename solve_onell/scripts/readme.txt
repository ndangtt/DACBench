The script plot.py is used to create a summary plot of an RL training given a log file.

It will extract all lines in the log files that containing "(training) Episode done" for "(evaluate) Episode done", extract stats from those lines and plot them together with the optimal values (read from the `.csv` files in folder `./data/`).

These log lines are from string `returned_info['msg']` in `OneLLEnv.step()` function: https://github.com/ndangtt/DACBench/blob/74968c05be3f026799f415cffb81ab543acdeaa2/dacbench/envs/onell_env.py#L520
These are only used when an episode is done.
