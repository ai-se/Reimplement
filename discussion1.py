from __future__ import division
from random import shuffle
from sklearn.tree import DecisionTreeRegressor
import scipy
import pandas as pd
import numpy as np


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


def policy1(scores, lives=3):
    """
    no improvement in last 3 runs
    """
    temp_lives = lives
    last = scores[0]
    for i,score in enumerate(scores):
        if i > 0:
            if temp_lives == 0:
                return i
            elif score >= last:
                temp_lives -= 1
                last = score
            else:
                temp_lives = lives
                last = score
    return -1


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
    print round(scipy.stats.stats.pearsonr(test_dependent, predicted)[0], 2), ",",
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
        policy_result = policy1(rank_diffs)
        # if policy_result != -1: break
        steps += 1
        sub_train_set.append(train_set[initial_size+steps])

    return sub_train_set


def draw_fig():
    y = [0.48 , 0.48 , 0.49 , 0.47 , 0.49 , 0.46 , 0.5 , 0.51 , 0.48 , 0.49 , 0.47 , 0.48 , 0.5 , 0.48 ,
         0.5 , 0.49 , 0.42 , 0.4 , 0.55 , 0.55 , 0.45 , 0.49 , 0.49 , 0.53 , 0.46 , 0.46 , 0.56 , 0.67 ,
         0.69 , 0.72 , 0.49 , 0.58 , 0.57 , 0.62 , 0.63 , 0.61 , 0.59 , 0.61 , 0.61 , 0.61 , 0.55 , 0.6 ,
         0.59 , 0.57 , 0.55 , 0.67 , 0.69 , 0.67 , 0.64 , 0.67 , 0.65 , 0.66 , 0.65 , 0.58 , 0.62 , 0.63 ,
         0.63 , 0.58 , 0.63 , 0.63 , 0.58 , 0.61 , 0.57 , 0.57 , 0.62 , 0.58 , 0.64 , 0.63 , 0.65 , 0.61 ,
         0.69 , 0.68 , 0.66 , 0.66 , 0.64 , 0.65 , 0.64 , 0.68 , 0.63 , 0.7 , 0.69 , 0.62 , 0.67 , 0.62 ,
         0.67 , 0.63 , 0.67 , 0.62 , 0.63 , 0.67 , 0.64 , 0.67 , 0.62 , 0.64 , 0.68 , 0.68 , 0.65 , 0.68 ,
         0.68 , 0.65 , 0.65 , 0.7 , 0.65 , 0.64 , 0.65 , 0.64 , 0.65 , 0.65 , 0.65 , 0.65 , 0.7 , 0.61 ,
         0.6 , 0.69 , 0.65 , 0.65 , 0.65 , 0.61 , 0.65 , 0.7 , 0.6 , 0.64 , 0.61 , 0.65 , 0.65 , 0.65 ,
         0.7 , 0.65 , 0.61 , 0.61 , 0.7 , 0.7 , 0.57 , 0.57 , 0.62 , 0.58 , 0.57 , 0.63 , 0.63 , 0.63 ,
         0.58 , 0.58 , 0.58 , 0.61 , 0.57 , 0.56 , 0.56 , 0.61 , 0.62 , 0.56 , 0.58 , 0.62 , 0.62 , 0.58 ,
         0.58 , 0.64 , 0.59 , 0.64 , 0.63 , 0.58 , 0.59 , 0.59 , 0.58 , 0.64 , 0.64 , 0.63 , 0.63 , 0.65 ,
         0.65 , 0.66 , 0.66 , 0.6 , 0.61 , 0.66 , 0.6 , 0.61 , 0.65 , 0.65 , 0.66 , 0.65 , 0.6 , 0.66 , 0.61 ,
         0.6 , 0.6 , 0.61 , 0.6 , 0.59 , 0.59 , 0.62 , 0.62 , 0.58 , 0.6 , 0.61 , 0.59 , 0.59 , 0.58 , 0.63 , 0.6 ,
         0.59 , 0.62 , 0.63 , 0.58 , 0.58 , 0.6 , 0.61 , 0.62 , 0.61 , 0.64 , 0.61 , 0.59 , 0.61 , 0.59 , 0.64 , 0.61 ,
         0.6 , 0.61 , 0.57 , 0.59 , 0.63 , 0.62 , 0.63 , 0.63 , 0.63 , 0.65 , 0.63 , 0.62 , 0.62 , 0.65 , 0.6 , 0.64 ,
         0.61 , 0.63 , 0.6 , 0.58 , 0.6 , 0.65 , 0.63 , 0.61 , 0.57 , 0.64 , 0.61 , 0.6 , 0.62 , 0.62 , 0.63 , 0.59 ,
         0.65 , 0.88 , 0.87 , 0.88 , 0.82 , 0.88 , 0.88 , 0.66 , 0.69 , 0.69 , 0.69 , 0.86 , 0.87 , 0.87 , 0.87 , 0.8 ,
         0.8 , 0.8 , 0.8 , 0.8 , 0.8 , 0.81 , 0.81 , 0.81 , 0.8 , 0.8 , 0.8 , 0.8 , 0.86 , 0.87 , 0.86 , 0.86 , 0.69 ,
         0.87 , 0.79 , 0.87 , 0.86 , 0.86 , 0.86 , 0.86 , 0.79 , 0.87 , 0.87 , 0.86 , 0.87 , 0.86 , 0.86 , 0.87 , 0.79,
         0.86 , 0.87 , 0.87 , 0.87 , 0.86 , 0.77 , 0.75 , 0.74 , 0.83 , 0.76 , 0.76 , 0.8 , 0.8 , 0.75 , 0.76 , 0.76 ,
         0.77 , 0.8 , 0.75 , 0.76 , 0.74 , 0.81 , 0.74 , 0.73 , 0.79 , 0.75 , 0.81 , 0.76 , 0.81 , 0.77 , 0.81 , 0.81 ,
         0.81 , 0.79 , 0.73 , 0.75 , 0.75 , 0.74 , 0.73 , 0.73 , 0.78 , 0.74 , 0.78 , 0.75 , 0.76 , 0.79 , 0.78 , 0.74 ,
         0.72 , 0.78 , 0.73 , 0.72 , 0.81 , 0.73 , 0.81 , 0.8 , 0.74 , 0.78 , 0.81 , 0.74 , 0.76 , 0.78 , 0.76 , 0.76 ,
         0.74 , 0.76 , 0.76 , 0.72 , 0.76 , 0.74 , 0.72 , 0.76 , 0.76 , 0.78 , 0.77 , 0.72 , 0.72 , 0.76 , 0.76 , 0.76 ,
         0.77 , 0.79 , 0.83 , 0.79 , 0.78 , 0.79 , 0.78 , 0.85 , 0.78 , 0.79 , 0.86 , 0.85 , 0.78 , 0.83 , 0.78 , 0.78 ,
         0.79 , 0.78 , 0.78 , 0.84 , 0.84 , 0.74 , 0.84 , 0.8 , 0.84 , 0.69 , 0.69 , 0.74 , 0.79 , 0.81 , 0.73 , 0.8 ,
         0.73 , 0.73 , 0.68 , 0.7 , 0.86 , 0.73 , 0.7 , 0.86 , 0.74 , 0.87 , 0.81 , 0.79 , 0.84 , 0.73 , 0.68 , 0.8 ,
         0.7 , 0.7 , 0.86 , 0.84 , 0.69 , 0.8 , 0.68 , 0.69 , 0.69 , 0.84 , 0.79 , 0.73 , 0.73 , 0.84 , 0.87 , 0.74 ,
         0.69 , 0.68 , 0.68 , 0.73 , 0.69 , 0.84 , 0.69 , 0.69 , 0.81 , 0.75 , 0.7 , 0.7 , 0.7 , 0.84 , 0.73 , 0.8 ,
         0.81 , 0.8 , 0.87 , 0.87 , 0.81 , 0.85 , 0.75 , 0.87 , 0.73 , 0.85 , 0.87 , 0.69 , 0.7 , 0.87 , 0.81 , 0.75 ,
         0.75 , 0.89 , 0.75 , 0.88 , 0.89 , 0.88 , 0.89 , 0.75 , 0.88 , 0.89 , 0.75 , 0.88 , 0.88 , 0.89 , 0.75 , 0.75 ,
         0.88 , 0.89 , 0.75 , 0.89 , 0.76 , 0.75 , 0.75 , 0.89 , 0.76 , 0.76 , 0.76 , 0.76 , 0.75 , 0.89 , 0.75 , 0.89 ,
         0.75 , 0.75 , 0.89 , 0.76 , 0.89 , 0.89 , 0.75 , 0.75 , 0.89 , 0.75 , 0.75 , 0.75 , 0.89 , 0.89 , 0.89 , 0.89 ,
         0.89 , 0.89 , 0.89 , 0.76 , 0.89 , 0.89 , 0.75 , 0.75 , 0.89 , 0.75 , 0.93 , 0.82 , 0.82 , 0.94 , 0.82 , 0.82 ,
         0.82 , 0.93 , 0.82 , 0.93 , 0.93 , 0.93 , 0.94 , 0.93 , 0.93 , 0.82 , 0.82 , 0.93 , 0.82 , 0.94 , 0.83 , 0.81 ,
         0.81 , 0.81 , 0.81 , 0.81 , 0.81 , 0.81 , 0.81 , 0.81 , 0.81 , 0.81 , 0.81 , 0.81 , 0.81 , 0.81 , 0.81 , 0.81 ,
         0.81 , 0.81 , 0.81 , 0.81 , 0.94 , 0.83 , 0.83 , 0.83 , 0.82 , 0.83 , 0.83 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 ,
         0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 ,
         0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 ,
         0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.93 , 0.93 , 0.93 , 0.93 , 0.93 , 0.93 , 0.94 , 0.94 ,
         0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 ,
         0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 ,
         0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 ,
         0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 ,
         0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 ,
         0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 ,
         0.95 , 0.95 , 0.95 , 0.95 , 0.94 , 0.95 , 0.94 , 0.94 , 0.94 , 0.94 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 ,
         0.95 , 0.94 , 0.94 , 0.94 , 0.95 , 0.95 , 0.94 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.94 , 0.94 ,
         0.95 , 0.94 , 0.94 , 0.94 , 0.94 , 0.94 , 0.95 , 0.94 , 0.95 , 0.95 , 0.95 , 0.95 , 0.94 , 0.94 , 0.95 , 0.95 ,
         0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 ,
         0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.94 , 0.95 , 0.95 , 0.94 , 0.95 ,
         0.95 , 0.94 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 ,
         0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.96 , 0.95 ,
         0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.96 , 0.95 , 0.95 , 0.95 , 0.95 ,
         0.96 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 ,
         0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.94 , 0.95 , 0.94 , 0.94 ,
         0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.94 , 0.94 , 0.94 , 0.95 , 0.94 , 0.94 , 0.94 , 0.95 , 0.94 , 0.94 , 0.95 ,
         0.95 , 0.94 , 0.95 , 0.94 , 0.95 , 0.95 , 0.95 , 0.94 , 0.94 , 0.95 , 0.95 ]
    x = range(1, len(y)+1)

    import matplotlib.pyplot as plt
    plt.plot(x, y, c='r')
    plt.xlabel("Size of the training set")
    plt.ylabel("Pearsson's Correlation")
    # plt.title("Apac")
    plt.savefig('figure7.eps')

from os import listdir
def run_experiment():
    datafolder = "./Data/"
    files = [datafolder + f for f in listdir(datafolder)]
    for file in files:
        print file
        mres = []
        sizes = []
        wins = 0
        lost = 0
        for _ in xrange(1):
            print "+ ",
            datasets = split_data(file)
            train_set = datasets[0]
            validation_set = datasets[1]
            test_set = datasets[2]
            print file,
            wrapper_rank_progressive(train_set, validation_set)
            print

if __name__ == "__main__":
    draw_fig()
    # run_experiment()