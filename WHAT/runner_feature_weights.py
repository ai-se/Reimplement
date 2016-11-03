from __future__ import division
from Methods.what import what
from Methods.guo_random import guo_random
from Utilities.model import generate_model
from Utilities.normalize import do_normalize_min_max, do_normalize_zscore
from Utilities.feature_weighting import add_feature_weights
import numpy as np
import pandas as pd
import random, sys, os
from sklearn.tree import DecisionTreeRegressor


def experiment2(filename, ret_val):
    mmre = []
    for _ in xrange(10):
        method = guo_random(filename)
        training_data, testing_data = method.generate_test_data(ret_val)
        assert(len(training_data[0]) == len(training_data[-1])), "Something is wrong"
        assert(len(testing_data[0]) == len(testing_data[-1])), "Something is wrong"
        mmre.append(generate_model(training_data, testing_data))

    print round(np.mean(mmre) * 100, 3), round(np.std(mmre) * 100, 3),


def experiment1(filename, normalize=None, feature_weights=False, no_of_trees=20, dist=1.5):
    if normalize is not None:
        name = filename.split('/')[-1]
        norm_filename = "./NData/zscore/norm_" + name

    mmre = []
    evals = []
    trains = []
    for _ in xrange(10):
        # print ". ",
        # sys.stdout.flush()
        _, extra_evals = add_feature_weights(norm_filename)
        trains.append(len(extra_evals[0]))
        # build bagging trees
        trees = [DecisionTreeRegressor() for _ in xrange(no_of_trees)]
        for i in xrange(no_of_trees):
            indexes = np.random.choice(range(len(extra_evals[0])), len(extra_evals[0]))
            indep_data_al = [extra_evals[0][index] for index in indexes]
            dep_data_al = [extra_evals[1][index] for index in indexes]
            assert(len(indep_data_al) == len(dep_data_al)), "something is wrong"
            trees[i].fit(indep_data_al, dep_data_al)

        content = pd.read_csv(filename)
        indexes = range(len(content))
        random.shuffle(indexes)

        cal_indep_train = []
        cal_dep_train = []
        training_indexes = indexes[:int(len(content) * 0.5)]
        testing_indexes = indexes[int(len(content) * 0.5):]

        training_data = content.ix[training_indexes]
        testing_data = content.ix[testing_indexes]

        indep_cols = [c for c in content.columns if "$<" not in c]
        dep_cols = [c for c in content.columns if "$<" in c]
        indep_training_data = training_data[indep_cols]
        dep_training_data = training_data[dep_cols]

        indep_testing_data = testing_data[indep_cols]
        dep_testing_data = testing_data[dep_cols[-1]].tolist()

        for indep_i in xrange(len(indep_training_data)):
            train_ind = indep_training_data.iloc[indep_i]
            train_dep = dep_training_data.iloc[indep_i]
            returns = []
            for tr in xrange(no_of_trees):
                returns.append(trees[tr].predict(train_ind).tolist())
            returns = [ret[-1] for ret in returns]
            measure = (np.percentile(returns, 75) - np.percentile(returns, 25))/np.mean(returns)
            # print returns, measure
            if measure >= dist:
                cal_indep_train.append(train_ind)
                cal_dep_train.append(train_dep)

        cal_indep_train.extend(indep_data_al)
        cal_dep_train.extend(dep_data_al)
        if len(cal_indep_train) == 0:return
        model = DecisionTreeRegressor()
        model.fit(cal_indep_train, cal_dep_train)

        predicted = model.predict(indep_testing_data)
        mre = []
        for actual, predict in zip(dep_testing_data, predicted):
            if abs(actual) != 0:
                mre.append(abs(actual - predict)/abs(actual))
        mmre.append(np.mean(mre) * 100)
        evals.append(len(cal_indep_train))
    print round(np.mean(mmre), 2), round(np.std(mmre), 2), #, round(np.mean(evals), 2), round(np.std(evals),3)
    return int(np.mean(evals))

if __name__ == "__main__":
    files = ["./Data/"+f for f in os.listdir("./Data/") if ".csv" in f]
    # files = ["./Data/Apache_AllMeasurements.csv"]
    # import pdb
    # pdb.set_trace()
    # for file in files:
    #     print file
    #     dists = [0.05 * i for i in xrange(1, 20)]
    #     for dist in dists:
    #         experiment1(file, normalize=do_normalize_zscore, feature_weights=True, dist=dist)
    #     print

    for file in files:
        print file,
        ret = experiment1(file, dist=0.4)
        experiment2(file, ret)
        print