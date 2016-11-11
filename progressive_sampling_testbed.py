from __future__ import division
import pandas as pd
import sys
from os import listdir
from random import shuffle
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import time
import matplotlib.pyplot as plt



class solution_holder:
    def __init__(self, id, decisions, objective, rank):
        self.id = id
        self.decision = decisions
        self.objective = objective
        self.rank = rank


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
                                       sortpdcontent.iloc[c][depcolumns].tolist(),
                                       c
                                       )
                       )

    shuffle(content)
    indexes = range(len(content))
    train_indexes, validation_indexes, test_indexes = indexes[:int(0.4*len(indexes))], indexes[int(.4*len(indexes)):int(.6*len(indexes))],  indexes[int(.6*len(indexes)):]
    assert(len(train_indexes) + len(validation_indexes) + len(test_indexes) == len(indexes)), "Something is wrong"
    train_set = [content[i] for i in train_indexes]
    validation_set = [content[i] for i in validation_indexes]
    test_set = [content[i] for i in test_indexes]

    return [train_set, validation_set, test_set]


def mre_progressive(train, test, threshold):
    train_independent = [t.decision for t in train]
    train_dependent = [t.objective[-1] for t in train]

    test_independent = [t.decision for t in test]
    test_dependent = [t.objective[-1] for t in test]

    model = DecisionTreeRegressor()
    model.fit(train_independent, train_dependent)
    predicted = model.predict(test_independent)

    mre = []
    for org, pred in zip(test_dependent, predicted):
        mre.append(abs(org - pred)/ abs(org))
    if np.mean(mre) <= threshold: return True
    else: return False


def rank_progressive(train, test, threshold=4):
    train_independent = [t.decision for t in train]
    train_dependent = [t.objective[-1] for t in train]

    sorted_test = sorted(test, key=lambda x: x.objective[-1])
    for r,st in enumerate(sorted_test): st.rank = r
    test_independent = [t.decision for t in sorted_test]
    test_dependent = [t.objective[-1] for t in sorted_test]

    model = DecisionTreeRegressor()
    model.fit(train_independent, train_dependent)
    predicted = model.predict(test_independent)
    predicted_id = [[i,p] for i,p in enumerate(predicted)]
    predicted_sorted = sorted(predicted_id, key=lambda x: x[-1])
    # assigning predicted ranks
    predicted_rank_sorted = [[p[0], p[-1], i] for i,p in enumerate(predicted_sorted)]
    rank_diffs = [abs(p[0] - p[-1]) for p in predicted_rank_sorted]
    return np.mean(rank_diffs)
    # if np.mean(rank_diffs) <= threshold:
    #     return True
    # else:
    #     return False


def progressive(filename, train_set, validation_set, check_condition=rank_progressive):
    # initial_size = 10
    # training_indexes = range(len(train_set))
    # shuffle(training_indexes)
    # sub_train_set = [train_set[i] for i in training_indexes[:initial_size]]
    # steps = 0
    # while check_condition(sub_train_set, validation_set) is False and (initial_size+steps) < len(train_set) - 1:
    #     sys.stdout.flush()
    #     steps += 1
    #     sub_train_set.append(train_set[initial_size+steps])
    #
    # return sub_train_set

    ## Delete
    initial_size = 10
    training_indexes = range(len(train_set))
    shuffle(training_indexes)
    sub_train_set = [train_set[i] for i in training_indexes[:initial_size]]
    steps = 0
    results = []
    while (initial_size + steps) < len(train_set) - 1:
        results.append(check_condition(sub_train_set, validation_set))
        steps += 1
        sub_train_set.append(train_set[initial_size+steps])

    import pickle
    pickle.dump(results, open("./PickleLocker/" + filename.split("/")[-1].split('.')[0] + ".p", "w"))
    plt.plot(range(len(results)), results)
    plt.savefig("./Figs/" + filename.split("/")[-1].split('.')[0])
    plt.cla()


    # sys.stdout.flush()
#     steps += 1
#     sub_train_set.append(train_set[initial_size+steps])


if __name__ == "__main__":
    datafolder = "./Data/"
    files = [datafolder + f for f in listdir(datafolder)]
    for file in files:
        print file
        mres = []
        datasets = split_data(file)
        train_set = datasets[0]
        validation_set = datasets[1]
        test_set = datasets[2]
        sub_train_set = progressive(file, train_set, validation_set)


        # for _ in xrange(20):
        #     print "+ ",
        #     datasets = split_data(file)
        #     train_set = datasets[0]
        #     validation_set = datasets[1]
        #     test_set = datasets[2]
        #     sub_train_set = progressive(train_set, validation_set)
        #
        #     # Test data
        #     train_independent = [t.decision for t in sub_train_set]
        #     train_dependent = [t.objective[-1] for t in sub_train_set]
        #
        #     test_independent = [t.decision for t in test_set]
        #     test_dependent = [t.objective[-1] for t in test_set]
        #
        #     model = DecisionTreeRegressor()
        #     model.fit(train_independent, train_dependent)
        #     predicted = model.predict(test_independent)
        #
        #     mre = []
        #     for org, pred in zip(test_dependent, predicted):
        #         mre.append(abs(org - pred) / abs(org))
        #     mres.append(np.mean(mre))
        # print
        # print file, np.mean(mres), " | ", mres
