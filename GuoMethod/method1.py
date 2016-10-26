from __future__ import division
from sklearn.tree import DecisionTreeRegressor
from random import shuffle
import numpy as np
import os

def run_alg(filename):
    content = open(filename).readlines()[1:]
    N = len(content[0].split(",")) -1
    indexes = range(len(content))

    for count in xrange(1, 5):
        mres = []
        for rep in xrange(30):
            shuffle(indexes)
            train_indexes = indexes[:count*N]
            test_indexes = indexes[count*N:]

            train_independent = []
            train_dependent = []

            test_independent = []
            test_dependent = []

            for ti in train_indexes:
                train_independent.append(map(int, content[ti].split(',')[:-1]))
                train_dependent.append(float(content[ti].split(',')[-1]))

            assert(len(train_independent) == len(train_dependent)), "something wrong"

            for ti in test_indexes:
                test_independent.append(map(int, content[ti].split(',')[:-1]))
                test_dependent.append(float(content[ti].split(',')[-1]))

            assert(len(test_independent) == len(test_dependent)), "something wrong"

            model = DecisionTreeRegressor()
            model.fit(train_independent, train_dependent)
            predicted = model.predict(test_independent)

            mre = []
            for i,j in zip(test_dependent, predicted):
                mre.append(abs(i - j)/i)
            mres.append(np.mean(mre))

        print count*N, round(np.mean(mres)*100, 4)


if __name__ == "__main__":
    files = ["./Data/" + f for f in os.listdir("./Data/") if ".csv" in f]
    for file in files:
        print file
        run_alg(file)