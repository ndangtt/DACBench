import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import sys

def read_evals_returns(fn):
    lines = []
    with open(fn, 'rt') as f:
        for ln in f:
            if ('Env train' in ln) or ('Env test' in ln):
                lines.append(ln[:-1])
    evals_returns = []
    for ln in lines:
        ls = ln.split(';')
        n_evals = int(list(filter(lambda s: 'evals=' in s, ls))[0].split('=')[1])
        returns = float(list(filter(lambda s: 'R=' in s, ls))[0].split('=')[1])
        evals_returns.append((n_evals,returns))
    t = pd.DataFrame(evals_returns, columns=['n_evals','returns'])
    t = t.sort_values(by='n_evals')
    return t

def plot_evals_returns(t, plot_file):
    t1 = t.groupby('n_evals').mean().reset_index()
    print(t1)
    plt.figure(figsize=(20,10))
    sns.lineplot(data=t, x='n_evals', y='returns')
    plt.savefig(plot_file)

#fn = 'out-500-sol0.95-interval4096-minibatch200-varinit1-imp_minus_evals'
fn = sys.argv[1]
t = read_evals_returns(fn)
plot_evals_returns(t, fn + ".png")
