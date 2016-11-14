from __future__ import division
import pickle
import numpy as np
import matplotlib.pyplot as plt

dict_results = pickle.load(open("mre_results.p", "r"))
files = dict_results.keys()
for file in files:
    filename = file.split("/")[-1].split(".")[0]
    dumb_mres = []
    dumb_stds = []
    dumb_evals = []
    random_mres = []
    random_stds = []
    random_evals = []
    fractions = sorted(dict_results[file].keys())
    for fraction in fractions:
        algorithms = dict_results[file][fraction].keys()
        dumb_mres.append(np.mean(dict_results[file][fraction]['dumb']['mres'])*100)
        dumb_stds.append(np.std(dict_results[file][fraction]['dumb']['mres'])*100)
        random_mres.append(np.mean(dict_results[file][fraction]['random-progressive']['mres'])*100)
        random_stds.append(np.std(dict_results[file][fraction]['random-progressive']['mres'])*100)
        dumb_evals.append(np.std(dict_results[file][fraction]['dumb']['train_set_size']))
        random_evals.append(np.std(dict_results[file][fraction]['random-progressive']['train_set_size']))

    left, width = .53, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height

    f, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(fractions, dumb_mres, 'ko-', color='r')
    ax1.plot(fractions, random_mres, 'kx-', color='g')
    ax1.set_xlim(0,)
    ax1.set_title(filename)
    ax1.set_ylabel("MMRE")
    if np.mean(dumb_mres) > 200:
        ax1.set_yscale('log')


    ax2.plot(fractions, dumb_stds, 'ko-', color='r')
    ax2.plot(fractions, random_stds, 'kx-', color='g')
    ax2.set_xlim(0, )
    if np.mean(dumb_stds) > 200:
        ax2.set_yscale('log')
    ax2.set_ylabel("STDS")


    ax3.plot(fractions, dumb_evals, 'ko-', color='r')
    ax3.plot(fractions, random_evals, 'kx-', color='g')
    ax3.set_xlim(0, )
    ax3.set_ylabel("Evaluations")

    plt.figlegend([ax1.lines[0], ax1.lines[1]],
                  ["Dumb", "Random-Progressive"], frameon=True, loc='lower center',
                  bbox_to_anchor=(0.48, -0.005), fancybox=True, ncol=2)

    f.text(0.40, 0.05, 'Percentage of Data', va='center', fontsize=13)
    f.set_size_inches(6, 9)
    plt.ylim(0,)

    plt.legend(loc='best')
    plt.savefig('./figures/'+filename+".png", bbox_inches='tight', format='png')
