from __future__ import division
import pandas as pd
from os import listdir
from random import shuffle


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
    indexes = xrange(len(content))
    train_indexes, validation_indexes, test_indexes = indexes[:0.4*len(indexes)], indexes[.4*len(indexes):.6*len(indexes)],  indexes[.6*len(indexes):]
    assert(len(train_indexes) + len(validation_indexes) + len(test_indexes) == len(indexes)), "Something is wrong"
    train_set = [content[i] for i in train_indexes]
    validation_set = [content[i] for i in validation_indexes]
    test_set = [content[i] for i in test_indexes]

    return [train_set, validation_set, test_set]


def progressive(train_set, validation_set, check_condition):
    initial_size = 10
    training_indexes = range(len(train_set))
    shuffle(training_indexes)



if __name__ == "__main__":
    datafolder = "./Data/"
    files = [datafolder + f for f in listdir(datafolder)]
    for file in files:
        datasets = split_data(file)