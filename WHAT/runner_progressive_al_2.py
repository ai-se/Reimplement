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
import pickle


def experiment2(filename, ret_val):
    mmre = []
    for _ in xrange(10):
        method = guo_random(filename)
        training_data, testing_data = method.generate_test_data(ret_val)
        assert(len(training_data[0]) == len(training_data[-1])), "Something is wrong"
        assert(len(testing_data[0]) == len(testing_data[-1])), "Something is wrong"
        mmre.append(generate_model(training_data, testing_data))

    print round(np.mean(mmre) * 100, 3), round(np.std(mmre) * 100, 3),
    return [m*100 for m in mmre]


def get_trees(extra_evals,  no_of_trees=20):
    trees = [DecisionTreeRegressor() for _ in xrange(no_of_trees)]
    for i in xrange(no_of_trees):
        indexes = np.random.choice(range(len(extra_evals[0])), len(extra_evals[0]))
        indep_data_al = [extra_evals[0][index] for index in indexes]
        dep_data_al = [extra_evals[1][index] for index in indexes]
        assert (len(indep_data_al) == len(dep_data_al)), "something is wrong"
        trees[i].fit(indep_data_al, dep_data_al)
    return trees

def experiment1(filename, normalize=None, feature_weights=False, dist=1.5, no_of_trees=20):
    if normalize is not None:
        name = filename.split('/')[-1]
        norm_filename = "./NData/zscore/norm_" + name

    mmre = []
    evals = []
    trains = []
    for _ in xrange(10):
        print ". ",
        sys.stdout.flush()
        filename, extra_evals = add_feature_weights(norm_filename)
        trains.append(len(extra_evals[0]))
        # build bagging trees


        content = pd.read_csv(filename)
        indexes = range(len(content))
        random.shuffle(indexes)

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
            # print len(extra_evals[0])
            trees = get_trees(extra_evals, no_of_trees)
            train_ind = indep_training_data.iloc[indep_i]
            train_dep = dep_training_data.iloc[indep_i]
            returns = []
            for tr in xrange(no_of_trees):
                returns.append(trees[tr].predict(train_ind).tolist())
            returns = [ret[-1] for ret in returns]
            measure = (np.percentile(returns, 75) - np.percentile(returns, 25))/np.mean(returns)
            if measure >= dist:
                extra_evals[0].append(train_ind)
                extra_evals[1].append(train_dep)

        model = DecisionTreeRegressor()
        model.fit(extra_evals[0], extra_evals[1])

        predicted = model.predict(indep_testing_data)
        mre = []
        for actual, predict in zip(dep_testing_data, predicted):
            if abs(actual) != 0:
                mre.append(abs(actual - predict)/abs(actual))
        mmre.append(np.mean(mre) * 100)
        evals.append(len(extra_evals[0]))
    print dist, round(np.mean(mmre), 2), round(np.std(mmre), 2), #round(np.mean(evals), 2), round(np.std(evals),3)
    return mmre, evals

if __name__ == "__main__":
    files = ['./Data/rs-6d-c3_obj1.csv', './Data/rs-6d-c3_obj2.csv', './Data/sol-6d-c2-obj1.csv', './Data/sol-6d-c2-obj2.csv', './Data/sort_256_obj2.csv', './Data/SQL_AllMeasurements.csv']
    results = {}
    for file in files:
        results[file] = {}
        print file,
        for dist in [0.05*i for i in xrange(1, 20)]:
            results[file][str(dist)] = {}
            al_mmre, al_evals = experiment1(file, normalize=do_normalize_zscore, dist=dist)
            ret = int(np.mean(al_evals))
            random_mmre = experiment2(file, ret)
            print
            results[file][str(dist)]["al_mmre"] = al_mmre
            results[file][str(dist)]["al_evals"] = al_evals
            results[file][str(dist)]["random_mmre"] = random_mmre
    pickle.dump(results, open("/Users/viveknair/GIT/Reimplement/WHAT/progressive_sampling/data/save2.p", "wb"))