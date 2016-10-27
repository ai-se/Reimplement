from __future__ import division
from sklearn.tree import DecisionTreeRegressor
from random import shuffle
import numpy as np
import os

def run_alg(filename):
    print filename,
    content = open(filename).readlines()[1:]
    N = len(content[0].split(",")) -1
    indexes = range(len(content))

    for count in xrange(1,10):
            mres = []
            for rep in xrange(30):
                shuffle(indexes)
                break_point = int(0.05 * count * len(content))
                train_indexes = indexes[:break_point]
                test_indexes = indexes[break_point:]

                train_independent = []
                train_dependent = []

                test_independent = []
                test_dependent = []

                for ti in train_indexes:
                    train_independent.append(map(float, content[ti].split(',')[:-1]))
                    train_dependent.append(float(content[ti].split(',')[-1]))

                assert(len(train_independent) == len(train_dependent)), "something wrong"

                for ti in test_indexes:
                    test_independent.append(map(float, content[ti].split(',')[:-1]))
                    test_dependent.append(float(content[ti].split(',')[-1]))


                assert(len(test_independent) == len(test_dependent)), "something wrong"

                model = DecisionTreeRegressor()
                model.fit(train_independent, train_dependent)
                predicted = model.predict(test_independent)

                mre = []
                for i,j in zip(test_dependent, predicted):
                    # print i, j
                    if i == 0:
                        continue

                    mre.append(abs(i - j)/abs(i))
                mres.append(np.mean(mre))

            print round(np.mean(mres)*100, 4),


if __name__ == "__main__":
    files = [
        # "AJStats.csv",
        # "Apache_AllMeasurements.csv",
        # "BDBC_AllMeasurements.csv",
        # "BDBJ_AllMeasurements.csv",
        # "clasp.csv",
        # "Dune.csv",
        # "EPL.csv",
        # "HSMGP_num.csv",
        # "Hipacc.csv",
        # "LLVM_AllMeasurements.csv",
        # "LinkedList.csv",
        # "PKJab.csv",
        # "SQL_AllMeasurements.csv",
        # "WGet.csv",
        # "X264_AllMeasurements.csv",
        # "ZipMe.csv",
        # "clasp.csv",
        # "lrzip.csv",
        # "x264.csv",
        "noc_CM_log_obj1.csv",
        "noc_CM_log_obj2.csv",
        "rs-6d-c3_obj1.csv",
        "rs-6d-c3_obj2.csv",
        "sol-6d-c2-obj1.csv",
        "sol-6d-c2-obj2.csv",
        "sort_256_obj1.csv",
        "sort_256_obj2.csv",
        "wc+rs-3d-c4-obj1.csv",
        "wc+rs-3d-c4-obj2.csv",
        "wc+sol-3d-c4-obj1.csv",
        "wc+sol-3d-c4-obj2.csv",
        "wc+wc-3d-c4-obj1.csv",
        "wc+wc-3d-c4-obj2.csv",
        "wc-3d-c4_obj1.csv",
        "wc-3d-c4_obj2.csv",
        "wc-5d-c5_obj1.csv",
        "wc-5d-c5_obj2.csv",
        "wc-6d-c1-obj1.csv",
        "wc-6d-c1-obj2.csv",
        "wc-c1-3d-c1-obj1.csv",
        "wc-c1-3d-c1-obj2.csv",
        "wc-c3-3d-c1-obj1.csv",
        "wc-c3-3d-c1-obj2.csv"

    ]
    for file in files:
        print file,
        run_alg("./Data/" + file)
        print