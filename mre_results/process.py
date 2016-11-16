from __future__ import division
import pickle
import numpy as np
import matplotlib.pyplot as plt

dict_results = pickle.load(open("merged.p", "r"))
files = dict_results.keys()
for file in files:
    print file
    filename = file.split("/")[-1].split(".")[0]
    dumb_mres = []
    dumb_stds = []
    dumb_evals = []
    dumb_ranks = []


    random_mres = []
    random_stds = []
    random_evals = []
    random_ranks = []

    atri_mres = []
    atri_stds = []
    atri_evals = []
    atri_ranks = []

    fractions = sorted(dict_results[file].keys())
    for fraction in fractions:
        algorithms = dict_results[file][fraction].keys()
        dumb_mres.append(np.mean(dict_results[file][fraction]['dumb']['mres'])*100)
        dumb_stds.append(np.std(dict_results[file][fraction]['dumb']['mres'])*100)

        random_mres.append(np.mean(dict_results[file][fraction]['random-progressive']['mres'])*100)
        random_stds.append(np.std(dict_results[file][fraction]['random-progressive']['mres'])*100)

        atri_mres.append(np.mean(dict_results[file][fraction]['random-projective']['mres'])*100)
        atri_stds.append(np.std(dict_results[file][fraction]['random-projective']['mres'])*100)

        dumb_evals.append(np.median(dict_results[file][fraction]['dumb']['train_set_size']))
        random_evals.append(np.median(dict_results[file][fraction]['random-progressive']['train_set_size']))
        atri_evals.append(np.median(dict_results[file][fraction]['random-projective']['train_set_size']))

        dumb_ranks.append(np.median(dict_results[file][fraction]['dumb']['min_rank']))
        random_ranks.append(np.median(dict_results[file][fraction]['random-progressive']['min_rank']))
        atri_ranks.append(np.median(dict_results[file][fraction]['random-projective']['min_rank']))


    left, width = .53, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height

    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.plot(fractions, dumb_mres, 'ko-', color='r')
    ax1.plot(fractions, random_mres, 'kx-', color='g')
    ax1.plot(fractions, atri_mres, 'kv-', color='b')

    if -1 in atri_mres:
        tf = []
        ava = []
        for i,j in zip(fractions, atri_mres):
            if j != -1:
                tf.append(i)
                ava.append(j)
        ax1.plot(tf, ava, 'kv-', color='b')
    else:
        ax1.plot(fractions, atri_mres, 'kv-', color='b')

    ax1.set_xlim(0,)
    ax1.set_title(filename)
    ax1.set_ylabel("MMRE")
    if np.mean(dumb_mres) > 200:
        ax1.set_yscale('log')

    ax2.plot(fractions, dumb_stds, 'ko-', color='r')
    ax2.plot(fractions, random_stds, 'kx-', color='g')
    if -1 in atri_stds:
        tf = []
        ava = []
        for i,j in zip(fractions, atri_stds):
            if j != -1:
                tf.append(i)
                ava.append(j)
        ax2.plot(tf, ava, 'kv-', color='b')
    else:
        ax2.plot(fractions, atri_stds, 'kv-', color='b')

    ax2.set_xlim(0, )
    if np.mean(dumb_stds) > 200:
        ax2.set_yscale('log')
    ax2.set_ylabel("STDS")

    ax3.plot(fractions, dumb_ranks, 'ko-', color='r')
    ax3.plot(fractions, random_ranks, 'kx-', color='g')
    if -1 in atri_ranks:
        tf = []
        ava = []
        for i,j in zip(fractions, atri_ranks):
            if j != -1:
                tf.append(i)
                ava.append(j)
        ax3.plot(tf, ava, 'kv-', color='b')
    else:
        ax3.plot(fractions, atri_ranks, 'kv-', color='b')
    ax3.set_xlim(0, )
    ax3.set_ylim(-0.5, max(max(dumb_ranks+random_ranks+atri_ranks)*1.5, 1))
    ax3.set_ylabel("Min Ranks")

    ax4.plot(fractions, dumb_evals, 'ko-', color='r')
    ax4.plot(fractions, random_evals, 'kx-', color='g')
    if -1 in atri_evals:
        tf = []
        ava = []
        for i,j in zip(fractions, atri_evals):
            if j != -1:
                tf.append(i)
                ava.append(j)
        ax4.plot(tf, ava, 'kv-', color='b')
    else:
        ax4.plot(fractions, atri_evals, 'kv-', color='b')
    ax4.set_xlim(0, )
    ax4.set_ylim(-0.5, max(dumb_evals+random_evals+atri_evals)*1.1)
    ax4.set_ylabel("Evaluations")

    plt.figlegend([ax1.lines[0], ax1.lines[1], ax1.lines[2]],
                  ["Dumb", "Random-Progressive", "Random-Projective"], frameon=True, loc='lower center',
                   bbox_to_anchor=(0.6, -0.005),fancybox=True, ncol=3)

    f.text(0.30, 0.05, 'Percentage of Data', va='center', fontsize=13)
    f.set_size_inches(6, 9)
    plt.ylim(0,)

    plt.legend(loc='best')
    plt.savefig('./figs/'+filename+".png", bbox_inches='tight', format='png')
