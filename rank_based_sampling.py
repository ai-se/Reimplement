from __future__ import division
import pandas as pd
from os import listdir
from random import shuffle
from sklearn.tree import DecisionTreeRegressor
from policies import policy
import numpy as np


class solution_holder:
    def __init__(self, id, decisions, objective, rank):
        self.id = id
        self.decision = decisions
        self.objective = objective
        self.rank = rank


def split_data(filename, fraction):
    """
    This method is used to split the data in the filename into training (40% or fraction), validation (20%) and testing (rest)
    :param filename: Filename of the csv file containing the configuration space
    :param fraction: Size of the training set.
    :return: list of list containing training, validation and testing set.
    """
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
    train_indexes, validation_indexes, test_indexes = indexes[:int(fraction*len(indexes))], indexes[int(fraction*len(indexes)):int((fraction+0.2)*len(indexes))],  indexes[int((fraction+0.2)*len(indexes)):]
    assert(len(train_indexes) + len(validation_indexes) + len(test_indexes) == len(indexes)), "Something is wrong"
    train_set = [content[i] for i in train_indexes]
    validation_set = [content[i] for i in validation_indexes]
    test_set = [content[i] for i in test_indexes]

    return [train_set, validation_set, test_set]


def rank_progressive(train, test):
    train_independent = [t.decision for t in train]
    train_dependent = [t.objective[-1] for t in train]

    sorted_test = sorted(test, key=lambda x: x.objective[-1])
    for r,st in enumerate(sorted_test): st.rank = r
    test_independent = [t.decision for t in sorted_test]

    model = DecisionTreeRegressor()
    model.fit(train_independent, train_dependent)
    predicted = model.predict(test_independent)
    predicted_id = [[i,p] for i,p in enumerate(predicted)]
    predicted_sorted = sorted(predicted_id, key=lambda x: x[-1])
    # assigning predicted ranks
    predicted_rank_sorted = [[p[0], p[-1], i] for i,p in enumerate(predicted_sorted)]
    rank_diffs = [abs(p[0] - p[-1]) for p in predicted_rank_sorted]
    return np.mean(rank_diffs)


def wrapper_rank_progressive(train_set, validation_set):
    initial_size = 10
    training_indexes = range(len(train_set))
    shuffle(training_indexes)
    sub_train_set = [train_set[i] for i in training_indexes[:initial_size]]
    steps = 0
    rank_diffs = []
    while (initial_size+steps) < len(train_set) - 1:
        rank_diffs.append(rank_progressive(sub_train_set, validation_set))
        policy_result = policy(rank_diffs)
        if policy_result != -1: break
        steps += 1
        sub_train_set.append(train_set[initial_size+steps])

    return sub_train_set


def find_lowest_rank(train, test):
    # Test data
    train_independent = [t.decision for t in train]
    train_dependent = [t.objective[-1] for t in train]

    sorted_test = sorted(test, key=lambda x: x.objective[-1])
    for r, st in enumerate(sorted_test): st.rank = r
    test_independent = [t.decision for t in sorted_test]

    model = DecisionTreeRegressor()
    model.fit(train_independent, train_dependent)
    predicted = model.predict(test_independent)

    predicted_id = [[i, p] for i, p in enumerate(predicted)]
    predicted_sorted = sorted(predicted_id, key=lambda x: x[-1])
    # assigning predicted ranks
    predicted_rank_sorted = [[p[0], p[-1], i] for i,p in enumerate(predicted_sorted)]
    select_few = predicted_rank_sorted[:10]
    return [sf[0] for sf in select_few]


def find_mre(train, test):
    # Test data
    train_independent = [t.decision for t in train]
    train_dependent = [t.objective[-1] for t in train]

    test_independent = [t.decision for t in test]
    test_dependent = [t.objective[-1] for t in test]

    model = DecisionTreeRegressor()
    model.fit(train_independent, train_dependent)
    predicted = model.predict(test_independent)

    mre = np.mean([abs(act-pred)/abs(act) for act, pred in zip(test_dependent, predicted) if act != 0])
    return mre


if __name__ == "__main__":
    """This program is used to execute rank-based methods"""
    datafolder = "./Data/"
    files = [datafolder + f for f in listdir(datafolder) if ".csv" in f]
    fractions = [0.4]
    results = {}
    for file in files:
        print file,
        results[file] = {}
        for fraction in fractions:
            print fraction,

            results[file][fraction] = {}
            results[file][fraction]["rank-based"] = {}
            results[file][fraction]["rank-based"]["mres"] = []
            results[file][fraction]["rank-based"]["train_set_size"] = []
            results[file][fraction]["rank-based"]["min_rank"] = []

            for _ in xrange(20):
                print " . ",
                datasets = split_data(file, fraction)

                train_set = datasets[0]
                validation_set = datasets[1]
                test_set = datasets[2]

                sub_train_set_rank = wrapper_rank_progressive(train_set, validation_set)
                lowest_rank = find_lowest_rank(sub_train_set_rank, test_set)
                mre = find_mre(sub_train_set_rank, test_set)
                results[file][fraction]["rank-based"]["mres"].append(mre)
                results[file][fraction]["rank-based"]["train_set_size"].append(len(sub_train_set_rank))
                results[file][fraction]["rank-based"]["min_rank"].append(min(lowest_rank))
            print
            print "Rank Difference: ", results[file][fraction]["rank-based"]["min_rank"]

    import pickle
    pickle.dump(results, open('PickleLocker/rank_based.p', 'w'))



