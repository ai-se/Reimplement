from __future__ import division
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from os import listdir
import pandas as pd
from random import shuffle

data_folder = "./Data/"

def mre_progressive(train, test):
    train_independent = [t.decision for t in train]
    train_dependent = [t.objective[-1] for t in train]
    # print len(train_independent)

    test_independent = [t.decision for t in test]
    test_dependent = [t.objective[-1] for t in test]

    model = DecisionTreeRegressor()
    model.fit(train_independent, train_dependent)
    predicted = model.predict(test_independent)

    mre = []
    for org, pred in zip(test_dependent, predicted):
        if org != 0:
            mre.append(abs(org - pred)/ abs(org))
    return np.mean(mre)


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
    train_indexes, test_indexes = indexes[:int(0.5*len(indexes))], indexes[int(.5*len(indexes)):]
    assert(len(train_indexes) + len(test_indexes) == len(indexes)), "Something is wrong"
    train_set = [content[i] for i in train_indexes]
    test_set = [content[i] for i in test_indexes]

    return [train_set, test_set]


class solution_holder:
    def __init__(self, id, decisions, objective):
        self.id = id
        self.decision = decisions
        self.objective = objective


if __name__ == "__main__":
    filename = "./Data/Apache_AllMeasurements.csv"
    data = split_data(filename)
    mres = []
    for i in xrange(1, len(data[0])):
        mres.append([i, 100-(mre_progressive(data[0][:i], data[1]) * 100)])

    import matplotlib.pyplot as plt
    import numpy as np

    x_axis = [m[0] for m in mres]
    y_axis = [m[1] for m in mres]
    plt.plot(x_axis, y_axis, color='r')

    plt.xlabel('Size of Training Set (# count)')
    plt.ylabel('Accuracy (%)')
    # plt.title('About as simple as it gets, folks')
    # plt.grid(True)
    plt.savefig("figure3.png")
    plt.show()
