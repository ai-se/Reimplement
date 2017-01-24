from __future__ import division
from os import listdir
import pandas as pd
import numpy as np
from random import shuffle
from sklearn.tree import DecisionTreeRegressor
import matplotlib.lines as lines
from matplotlib.lines import Line2D

"""
This code is used to generate the figure 1 of the paper
"""

data_folder = "./Data/"


class solution_holder:
    def __init__(self, id, decisions, objective):
        self.id = id
        self.decision = decisions
        self.objective = objective


def split_data(filename):
    pdcontent = pd.read_csv(filename)
    indepcolumns = [col for col in pdcontent.columns if "$<" not in col]
    depcolumns = [col for col in pdcontent.columns if "$<" in col]
    sortpdcontent = pdcontent.sort(depcolumns[-1])
    content = list()
    for c in xrange(len(sortpdcontent)):
        content.append(solution_holder(
                                       c,
                                       sortpdcontent.iloc[c][indepcolumns].tolist(),
                                       sortpdcontent.iloc[c][depcolumns].tolist()
                                       )
                       )

    shuffle(content)
    indexes = range(len(content))
    train_indexes, test_indexes = indexes[:int(0.3*len(indexes))], indexes[int(.3*len(indexes)):]
    assert(len(train_indexes) + len(test_indexes) == len(indexes)), "Something is wrong"
    train_set = [content[i] for i in train_indexes]
    test_set = [content[i] for i in test_indexes]

    return [train_set, test_set]


def mre_progressive(train, test):
    train_independent = [t.decision for t in train]
    train_dependent = [t.objective[-1] for t in train]

    test_independent = [t.decision for t in test]
    test_dependent = [t.objective[-1] for t in test]

    model = DecisionTreeRegressor()
    model.fit(train_independent, train_dependent)
    predicted = model.predict(test_independent)

    mre = []
    for org, pred in zip(test_dependent, predicted):
        if org != 0:
            mre.append(abs(org - pred)/ abs(org))
    return round(np.mean(mre)* 100, 2)


def get_data():
    files = [data_folder + file for file in listdir(data_folder)]
    result = []
    for file in files:
        mres = []
        for _ in xrange(20):
            train, test = split_data(file)
            mres.append(mre_progressive(train, test))
        result.append([file, [np.mean(mres), np.std(mres)], len(train) + len(test)])
    return result

def draw_fig(data):
    import matplotlib.pyplot as plt
    data = sorted(data, key=lambda x: x[1])
    projects = ["SS"+str(i+1) for i,d in enumerate(data)]
    y_pos = [i*10 for i in np.arange(len(projects))]
    performance = [d[1][0] for d in data]

    plt.plot([-5, 215], [5, 5], 'k-', lw=2)

    plt.bar(y_pos[:4], performance[:4], align='center', alpha=0.5, width=8, color='green', label='< 5%', hatch='o')
    plt.bar(y_pos[4:13], performance[4:13], align='center', alpha=0.5, width=8, color='yellow', label='5% < x < 10%', hatch='O')
    plt.bar(y_pos[13:], performance[13:], align='center', alpha=0.5, width=8, color='red', label='>10%', hatch='.')
    plt.xticks(y_pos, projects, rotation='vertical')
    plt.yscale('log')
    plt.xlim(-15, 225)
    plt.ylabel('MMRE')

    # Now add the legend with some customizations.
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True, frameon=False)

    plt.savefig('figure1.eps')


def gather_data():
    data = get_data()
    for d in data: print d  # debug
    import pdb
    pdb.set_trace()
    import pickle
    pickle.dump(data, open('fig1.p', 'w'))
    print "Done"


def draw_figure():
    import pickle
    data = pickle.load(open('fig1.p', 'r'))
    draw_fig(data)

def misc():
    import pickle
    data = pickle.load(open('fig1.p', 'r'))
    data = sorted(data, key=lambda x:x[1][0])
    import pdb
    pdb.set_trace()

if __name__ == "__main__":
    # gather_data()
    draw_figure()
    # misc()