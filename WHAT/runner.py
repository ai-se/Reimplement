from __future__ import division
from Methods.what import what
from Methods.guo_random import guo_random
from Utilities.model import generate_model
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import scipy.stats as ss


def random_sampling(filename, ret_val):
    mmre = []

    for _ in xrange(10):
        method = guo_random(filename)
        training_data, testing_data = method.generate_test_data(ret_val)
        assert(len(training_data[0]) == len(training_data[-1])), "Something is wrong"
        assert(len(testing_data[0]) == len(testing_data[-1])), "Something is wrong"
        model, perf = generate_model(training_data, testing_data)
        return model, testing_data, perf


def number_of_lines(filename):
    import pandas as pd
    content = pd.read_csv(filename)
    return len(content)


class instance_holder:
    def __init__(self, id, decision, objective, pred, rank):
        self.id = id
        self.decision = decision
        self.objective = objective
        self.pred = pred
        self.rank = rank

if __name__ == "__main__":
    files = ["./Data/"+f for f in os.listdir("./Data/") if ".csv" in f]
    result = {}
    for file in files:
        result[file] = {}
        print file,
        no_of_configurations = number_of_lines(file)
        values = [int(0.01 * i  * no_of_configurations) for i in xrange(1, 51)]

        # This makes sure that the testing data is same for all values
        _, testing_data, _ = random_sampling(file, int(0.05*no_of_configurations))

        for val in values:
            result[file][val] = {}
            model, _, perf = random_sampling(file, val)
            rank = ss.rankdata(testing_data[1], method='min')
            predictions = model.predict(testing_data[0])
            instances = {}
            list_instances = []
            for i, pred in enumerate(predictions):
                instances[i] = instance_holder(i, testing_data[0][i], testing_data[1][i], pred, rank[i])
                list_instances.append(instance_holder(i, testing_data[0][i], testing_data[1][i], pred, rank[i]))

            # find corresponding dependent values of the top predicted values
            temp = sorted(list_instances, key=lambda x:x.pred)[:10]
            dep_values = [t.objective for t in temp]
            rank_values = [t.rank for t in temp]

            plt.title(file.split('/')[-1].split('.')[0] + '-' + str(val))

            plt.xlabel('Ranks')
            plt.ylabel('Performance Score')
            plt.plot(xrange(len(testing_data[1])), sorted(testing_data[1]), color='r')
            result[file][val]['mmre'] = round(perf, 3) * 100
            result[file][val]['min_rank'] = min(rank_values)
            result[file][val]['median_rank'] =  np.median(rank_values)

            print round(perf, 3) * 100, min(rank_values), np.median(rank_values),
        print
        import pickle

        pickle.dump(result, open("save.p", "wb"))
            # for i,j in zip(rank_values, dep_values):
            #     plt.scatter(i, j)
            # plt.show()
            # plt.cla()