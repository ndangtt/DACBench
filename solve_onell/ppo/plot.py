#!/home/nttd/anaconda3/envs/dacbench/bin/python


import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import os
import json

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
    exp_config_file = exp_dir + "/exp_config.json"   
    with open(exp_config_file, 'rt') as f:
        exp_conf = json.load(f)
    bench_config_file = exp_dir + '/bench_config.json' 
    with open(bench_config_file, 'rt') as f:
        bench_conf = json.load(f)

    
    # read txt log file
    log_txt = exp_dir + '/log.txt'
    with open(log_txt, 'rt') as f:
        ls_lines = [s[:-1] for s in f if 'Episode done:' in s]
    
    # training plot
    vals = [np.array([float(s.split('=')[1]) for s in line.split('Episode done:')[1].split('; ')])[[1,4,7,8,9,10]] for line in ls_lines if '(training)' in line]
    #print(np.array([float(s.split('=')[1]) for s in line.split('Episode done:')[1].split('; ')])[[1,4,7,8,9,10]])
    t = pd.DataFrame(vals, columns=['n','evals','lbd_min','lbd_max','lbd_mean','R'])
    ax1 = plt.subplot(nrows,ncols,plot_id_start)
    plt.plot(t[value_name])
    plt.title('training-' + value_name)
    y1_min = t[value_name].min()
    y1_max = t[value_name].max()

    # evaluation plot
    vals = [np.array([float(s.split('=')[1]) for s in line.split('Episode done:')[1].split('; ')])[[1,4,7,8,9,10]] for line in ls_lines if '(evaluate)' in line]
    t = pd.DataFrame(vals, columns=['n','evals','lbd_min','lbd_max','lbd_mean','R'])
    eval_episodes = int(exp_conf['eval_n_episodes'])
    n_rows = len(t.index)
    t['iteration'] = np.repeat(np.arange(np.ceil(n_rows/eval_episodes)), eval_episodes)[:n_rows]
    t1 = t.groupby(['n','iteration'])[value_name].agg([np.mean,np.std]).reset_index()
    ax2 = plt.subplot(nrows,ncols,plot_id_start+1)
    plt.plot(t1.iteration, t1['mean'])
    plt.fill_between(t1.iteration, t1['mean'] - t1['std'], t1['mean'] + t1['std'], alpha=0.2)
    plt.xlim([t1.iteration.min(),t1.iteration.max()])
    y2_min = (t1['mean'] - t1['std']).min()
    y2_max = (t1['mean'] - t1['std']).max()


    # plot the optimal
    opt_min = None
    opt_max = None
    if value_name in ['R','evals']:
        init_solution_ratio = None
        if ('init_solution_ratio' in bench_conf) and (bench_conf['init_solution_ratio'] != None):
            init_solution_ratio = float(bench_conf['init_solution_ratio'])
        if value_name == 'R':
            optim_value_name = bench_conf['reward_choice']
        elif value_name == 'evals':
            optim_value_name = 'n_evals'
        mean, std = read_optimal_results(t.n[0], init_solution_ratio, optim_value_name)
        plt.plot(t1.iteration, [mean]*len(t1.iteration), color='red')
        plt.fill_between(t1.iteration, [mean-std]*len(t1.iteration), [mean+std]*len(t1.iteration), color='red', alpha=0.2)
        opt_min = mean - std
        opt_max = mean + std

    plt.title('evaluation-'+value_name)

    # set ylim of the training and evaluation plots to be the same
    y_min = min(y1_min, y2_min)    
    y_max = max(y1_max, y2_max)
    if opt_min:
        y_min = min(y_min, opt_min)
        y_max = max(y_max, opt_max)
    ax1.set_ylim([y_min,y_max])
    ax2.set_ylim([y_min,y_max])
    
    #plt.show()
    #plt.savefig(exp_dir + '/plot-' + value_name + '.png')




def main():
    exp_dir = sys.argv[1]
    while exp_dir[-1]=='/':
        exp_dir = exp_dir[:-1]
    value_names = ['evals','R','lbd_mean', 'lbd_max']
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
