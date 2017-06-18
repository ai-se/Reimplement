from __future__ import division
from random import shuffle
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import sys


def get_number_of_points(filename):
    counts = {}
    lines = open(filename).readlines()
    for line in lines:
        if ">>>>" in line:
            temp = line.replace(">>>>", "").strip().split(" ")
            counts[temp[0]] = float(temp[-1])
    return counts


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

if __name__ == "__main__":
    """
    This is to run projective sampling. 
    atri_results.txt contains the number of measurement required to achieve 90\% of accuracy
    as predicted by the projective sampling method proposed in the paper.
    """
    results = {}
    counts = get_number_of_points("./atri_results.txt")
    for filename in counts.keys():
        print filename,
        fully_name = "./Data/" + filename
        results[fully_name] = {}
        results[fully_name][0.4] = {}
        results[fully_name][0.4]['projective'] = {}
        results[fully_name][0.4]['projective']['train_set_size'] = []
        results[fully_name][0.4]['projective']['mres'] = []
        results[fully_name][0.4]['projective']['min_rank'] = []
        for _ in range(20):
            print ". ",
            sys.stdout.flush()
            datasets = split_data(fully_name)
            train_set = datasets[0] + datasets[1]
            test_set = datasets[2]
            shuffle(train_set)
            sub_train_set = train_set[:int(counts[filename])]

            model = DecisionTreeRegressor()
            model.fit([t.decision for t in sub_train_set], [t.objective[-1] for t in sub_train_set])
            test_set = sorted(test_set, key=lambda x:x.objective[-1])
            test_set_independent = [t.decision for t in test_set]
            test_set_dependent = [t.objective[-1] for t in test_set]
            predicted = model.predict(test_set_independent)

            mre = np.mean([abs(pred-act)/abs(act) for pred, act in zip(predicted, test_set_dependent) if act != 0])

            predicted = [[i,p] for i,p in enumerate(predicted)]
            predicted = [[p[0],p[1], i] for i,p in enumerate(sorted(predicted, key=lambda x: x[1]))]
            select_few = [p[0] for p in predicted[:10]]

            results[fully_name][0.4]['projective']['train_set_size'].append(int(counts[filename]))
            results[fully_name][0.4]['projective']['mres'].append(mre)
            results[fully_name][0.4]['projective']['min_rank'].append(min(select_few))
        print
        print "Rank Difference: ", results[fully_name][0.4]['projective']['min_rank']

    import pickle
    pickle.dump(results, open('PickleLocker/projective_sampling.p', "w"))