import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def read_optimal_results(n, init_solution_ratio, prefix='./data/dyn_theory'):
    if init_solution_ratio:
        fn = prefix + '-' + str(init_solution_ratio) + '.csv'
    else:
        fn = prefix + '.csv'
    t = pd.read_csv(fn)
    ls = t[t.n==n].n_evals
    return ls.mean(), ls.std()


def plot_td3(config_file, exp_dir):
    plt.clf()

    with open(config_file, 'rt') as f:
        conf = yaml.load(f, Loader=yaml.Loader)
    
    log_txt = exp_dir + '/' + conf['expt']['project_name'] + '/' + conf['expt']['session_name'] + '/' + conf['expt']['experiment_name'] + '-0/log/log_file.txt'
    with open(log_txt, 'rt') as f:
        ls_lines = [s[:-1] for s in f if 'Episode done:' in s]
    
    # training plot
    vals = [np.array([float(s.split('=')[1]) for s in line.split('Episode done:')[1].split('; ')])[[1,4,7,8,9,10]] for line in ls_lines if '(training)' in line]
    #print(np.array([float(s.split('=')[1]) for s in line.split('Episode done:')[1].split('; ')])[[1,4,7,8,9,10]])
    t = pd.DataFrame(vals, columns=['n','evals','lbd_min','lbd_max','lbd_mean','R'])
    plt.subplot(1,2,1)
    plt.plot(t['evals'])
    plt.title('training plot')

    # evaluation plot
    vals = [np.array([float(s.split('=')[1]) for s in line.split('Episode done:')[1].split('; ')])[[1,4,7,8,9,10]] for line in ls_lines if '(evaluate)' in line]
    t = pd.DataFrame(vals, columns=['n','evals','lbd_min','lbd_max','lbd_mean','R'])
    if 'td3' in conf:
        eval_episodes = int(conf['td3']['eval_episodes'])
    else:
        eval_episodes = int(conf['eval']['test_episodes'])
    n_rows = len(t.index)
    t['iteration'] = np.repeat(np.arange(np.ceil(n_rows/eval_episodes)), eval_episodes)[:n_rows]
    t1 = t.groupby(['n','iteration']).evals.agg([np.mean,np.std]).reset_index()
    plt.subplot(1,2,2)
    plt.plot(t1.iteration, t1['mean'])
    plt.fill_between(t1.iteration, t1['mean'] - t1['std'], t1['mean'] + t1['std'], alpha=0.2)
    plt.xlim([t1.iteration.min(),t1.iteration.max()])

    # plot the optimal
    init_solution_ratio = None
    if ('init_solution_ratio' in conf['bench']) and (conf['bench']['init_solution_ratio'] != None):
        init_solution_ratio = float(conf['bench']['init_solution_ratio'])
    mean, std = read_optimal_results(t.n[0], init_solution_ratio)
    plt.plot(t1.iteration, [mean]*len(t1.iteration), color='red')
    plt.fill_between(t1.iteration, [mean-std]*len(t1.iteration), [mean+std]*len(t1.iteration), color='red', alpha=0.2)

    plt.title('evaluation plot')

    plt.show()




def main():
    parser = argparse.ArgumentParser("plot searl/td3 results on onell")
    parser.add_argument("--exp_type",type=str,choices=['searl','td3'])
    parser.add_argument("--config_file",type=str)
    parser.add_argument("--exp_dir",type=str)
    args = parser.parse_args()
    while True:
        plot_td3(args.config_file, args.exp_dir)
        time.sleep(1)

main()
