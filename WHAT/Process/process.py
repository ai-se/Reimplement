from __future__ import division
import pickle
import matplotlib.pyplot as plt


result = pickle.load(open("save.p", "rb"))
files = result.keys()
for file in files:
    vals = sorted(result[file].keys())
    mmres = [result[file][val]['mmre'] for val in vals]
    top_ranks = [result[file][val]['min_rank'] for val in vals]

    fig, ax1 = plt.subplots()
    ax1.plot(vals, mmres, 'b-')
    ax1.set_xlabel('Size of training set')
    ax1.set_ylabel('MMRE', color='b')
    ax1.set_ylim(0, 20)
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    ax2 = ax1.twinx()
    ax2.plot(vals, top_ranks, 'r')
    ax2.set_ylabel('Minimum Rank Found', color='r')
    ax2.set_ylim(0, 20)
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    plt.suptitle(file.split('/')[-1].split('.')[0])
    plt.savefig("./Figs/" + file.split('/')[-1].split('.')[0])
    plt.cla()
    plt.close()

    # import pdb
    # pdb.set_trace()
