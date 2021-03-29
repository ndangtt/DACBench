import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import sys

# plot average returns and n_evals during evaluations
def plot_eval_returns_and_evals(fn):
    rs = {'checkpoint':[], 'return':[], 'n_evals': []}
    cur_checkpoint = 1
    reading = False
    with open(fn, 'rt') as f:
        for ln in f:
            if 'Env test' in ln:
                reading = True
                ls = ln.split(";") 
                n_evals = int(list(filter(lambda s: 'evals=' in s, ls))[0].split('=')[1])
                returns = float(list(filter(lambda s: 'R=' in s, ls))[0].split('=')[1])
                rs['checkpoint'].append(cur_checkpoint)
                rs['return'].append(returns)
                rs['n_evals'].append(n_evals)
            elif ('evaluation' in ln) and reading:
                cur_checkpoint += 1
                reading = False
    t = pd.DataFrame.from_dict(rs)
    print(t)

    plt.figure(figsize=(20,10))
    sns.lineplot(data=t, x='checkpoint', y='return')
    plt.savefig(fn + "-returns.png")

    plt.figure(figsize=(20,10))
    sns.lineplot(data=t, x='checkpoint', y='n_evals')
    plt.savefig(fn + "-n_evals.png")


#fn = 'out-500-sol0.95-interval4096-minibatch200-varinit1-imp_minus_evals'
fn = sys.argv[1]
plot_eval_returns_and_evals(fn)
