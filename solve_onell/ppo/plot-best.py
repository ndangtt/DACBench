import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import sys
import glob
import os

# plot average returns and n_evals during evaluations
def plot_comparison():
    rl_dir = sys.argv[1]

    rs_onemax_dyn_theory_500_095 = [1470,1757,2158,1258,1814,1570,1438,1723,1714,1358,2990,1736,1049,1801,1501,1315,1417,1867,1629,964,1354,1318,2052,1459,1746,1468,2051,1589,1649,1541,1522,1444,1319,2299,1062,1501,1194,1829,1183,1667]
    t = pd.DataFrame.from_dict({'alg':['dyn_theory'] * len(rs_onemax_dyn_theory_500_095), 'n_evals': rs_onemax_dyn_theory_500_095})

    ls_files = glob.glob(rl_dir + '/out-*')
    for fn in ls_files:
        rs = []
        with open(fn, 'rt') as f:
            for ln in f:
                if 'Env test' in ln:
                    ls = ln.split(";") 
                    n_evals = int(list(filter(lambda s: 'evals=' in s, ls))[0].split('=')[1])
                    rs.append(n_evals)        
        t1 = pd.DataFrame.from_dict({'alg':[os.path.basename(fn).split('-')[1]] * len(rs), 'n_evals': rs})
        t = pd.concat([t,t1],axis=0)

    plt.figure(figsize=(20,10))
    sns.boxplot(data=t, x='alg', y='n_evals')
    plt.savefig(rl_dir.replace('output-','out-') + "-comparison.png")


#fn = 'out-500-sol0.95-interval4096-minibatch200-varinit1-imp_minus_evals'
plot_comparison()
