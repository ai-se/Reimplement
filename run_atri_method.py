from __future__ import division
from random import shuffle
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import sys, os


def get_number_of_points(filename):
    counts = {}
    lines = open(filename).readlines()
    for line in lines:
        if ">>>>" in line and "FAILED" not in line:
            temp = line.replace(">>>>", "").strip().split(" ")
            if temp[1] not in counts.keys():
                counts[temp[1]]={}
                counts[temp[1]][float(temp[0])] = temp[2]
            counts[temp[1]][float(temp[0])] = temp[2]
    return counts


class solution_holder:
    def __init__(self, id, decisions, objective, rank):
        self.id = id
        self.decision = decisions
        self.objective = objective
        self.rank = rank


def split_data(filename, fraction):
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

if __name__ == "__main__":
    name_replace = {
        "HSMGP.csv": "HSMGP_num.csv",
        "BDBC.csv": "BDBC_AllMeasurements.csv",
        "SQL.csv": "SQL_AllMeasurements.csv",
        "Apache.csv": "Apache_AllMeasurements.csv",

    }
    results = {}
    counts = get_number_of_points("./atri_results2.txt")

    for filename in counts.keys():
        results[filename] = {}
        if os.path.isfile("./Data/"+filename) is False:
            fully_name = "./Data/" + name_replace[filename]
        else:
            fully_name = "./Data/" + filename

        print filename,os.path.isfile(fully_name)

        fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        for fraction in fractions:
            if fraction not in counts[filename].keys():
                results[filename][fraction] = {}
                continue
            results[filename][fraction] = {}
            results[filename][fraction]['atri_train_set'] = []
            results[filename][fraction]['atri_min_mre'] = []
            results[filename][fraction]['atri_min_rank'] = []
            for _ in range(20):
                print " + ",
                sys.stdout.flush()
                datasets = split_data(fully_name, fraction)
                train_set = datasets[0]
                test_set = datasets[2]
                shuffle(train_set)
                sub_train_set = train_set[:int(counts[filename][fraction])]
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
                results[filename][fraction]['atri_train_set'].append(int(counts[filename][fraction]))
                results[filename][fraction]['atri_min_mre'].append(mre)
                results[filename][fraction]['atri_min_rank'].append(min(select_few))
                print int(counts[filename][fraction]), mre,min(select_few)

    import pickle
    pickle.dump(results, open('atri_result.p', "w"))