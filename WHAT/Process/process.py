from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt

file = "./active_learning.txt"

content = open(file).readlines()
dict = {}
for line in content:
    if "./Data" in line:
        name = line.strip().split('/')[-1].split('.')[0]
        dict[name] = []
    elif line != '\n':
        dict[name].append(map(float, line.strip().split(' ')))

for key in dict.keys():
    df =pd.DataFrame(dict[key])
    df.columns = ['measure', 'mmre', 'stdmre', 'meanEvals', 'stdEvals']
    fig, ax1 = plt.subplots()
    ax1.plot(df['measure'], df['mmre'], 'b-')
    ax1.set_xlabel('Measure')
    ax1.set_ylabel('MMRE', color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    ax2 = ax1.twinx()
    ax2.plot(df['measure'], df['meanEvals'], 'r')
    ax2.set_ylabel('Evals', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    plt.suptitle(key)
    plt.savefig("./Figs/" + key)

    # import pdb
    # pdb.set_trace()
