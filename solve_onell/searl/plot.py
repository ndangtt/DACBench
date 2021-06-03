#!/home/nttd/anaconda3/envs/dacbench/bin/python


import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import os

script_path = os.path.dirname(os.path.realpath(__file__))


def read_optimal_results(n, init_solution_ratio, value_name, prefix=script_path + '/data/dyn_theory'):
    if init_solution_ratio:
        fn = prefix + '-' + str(init_solution_ratio) + '.csv'
    else:
        fn = prefix + '.csv'
    t = pd.read_csv(fn)
    ls = t[t.n==n][value_name]
    return ls.mean(), ls.std()


def make_plot(exp_dir, value_name, nrows=1, ncols=2, plot_id_start=1):
    #plt.clf()

    # read configuration
    config_file = exp_dir + "/config/config.yml"   
    with open(config_file, 'rt') as f:
        conf = yaml.load(f, Loader=yaml.Loader)
    
    # read txt log file
    log_txt = exp_dir + '/log/log_file.txt'
    with open(log_txt, 'rt') as f:
        ls_lines = [s[:-1] for s in f if 'Episode done:' in s]
    
    # training plot
    vals = [np.array([float(s.split('=')[1]) for s in line.split('Episode done:')[1].split('; ')])[[1,4,7,8,9,10]] for line in ls_lines if '(training)' in line]
    #print(np.array([float(s.split('=')[1]) for s in line.split('Episode done:')[1].split('; ')])[[1,4,7,8,9,10]])
    t = pd.DataFrame(vals, columns=['n','evals','lbd_min','lbd_max','lbd_mean','R'])
    plt.subplot(nrows,ncols,plot_id_start)
    plt.plot(t[value_name])
    plt.title('training-' + value_name)

    # evaluation plot
    vals = [np.array([float(s.split('=')[1]) for s in line.split('Episode done:')[1].split('; ')])[[1,4,7,8,9,10]] for line in ls_lines if '(evaluate)' in line]
    t = pd.DataFrame(vals, columns=['n','evals','lbd_min','lbd_max','lbd_mean','R'])
    if 'td3' in conf:
        eval_episodes = int(conf['td3'].eval_episodes)
    else:
        eval_episodes = int(conf['eval'].test_episodes)
    n_rows = len(t.index)
    t['iteration'] = np.repeat(np.arange(np.ceil(n_rows/eval_episodes)), eval_episodes)[:n_rows]
    t1 = t.groupby(['n','iteration'])[value_name].agg([np.mean,np.std]).reset_index()
    plt.subplot(nrows,ncols,plot_id_start+1)
    plt.plot(t1.iteration, t1['mean'])
    plt.fill_between(t1.iteration, t1['mean'] - t1['std'], t1['mean'] + t1['std'], alpha=0.2)
    plt.xlim([t1.iteration.min(),t1.iteration.max()])

    # plot the optimal
    if value_name in ['R','evals']:
        init_solution_ratio = None
        if ('init_solution_ratio' in conf['bench'].__dict__) and (conf['bench'].init_solution_ratio != None):
            init_solution_ratio = float(conf['bench'].init_solution_ratio)
        if value_name == 'R':
            optim_value_name = conf['bench'].reward_choice
        elif value_name == 'evals':
            optim_value_name = 'n_evals'
        mean, std = read_optimal_results(t.n[0], init_solution_ratio, optim_value_name)
        plt.plot(t1.iteration, [mean]*len(t1.iteration), color='red')
        plt.fill_between(t1.iteration, [mean-std]*len(t1.iteration), [mean+std]*len(t1.iteration), color='red', alpha=0.2)

    plt.title('evaluation-'+value_name)

    #plt.show()
    #plt.savefig(exp_dir + '/plot-' + value_name + '.png')




def main():
    exp_dir = sys.argv[1]
    while exp_dir[-1]=='/':
        exp_dir = exp_dir[:-1]
    value_names = ['evals','R','lbd_mean']
    nrows = len(value_names)
    ncols = 2
    while True:
        plt.clf()
        plot_start_id = 1
        for value_name in value_names:
            make_plot(exp_dir, value_name, nrows, ncols, plot_start_id)
            plot_start_id += 2
        plt.tight_layout()
        plt.savefig(exp_dir + '/plot.png')
        plt.savefig('plot-' + os.path.basename(exp_dir) +'.png')
        #time.sleep(1)
        sys.exit(0)

main()
