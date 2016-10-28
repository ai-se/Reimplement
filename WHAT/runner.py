from __future__ import division
from Methods.what import what
from Methods.guo_random import guo_random
from Utilities.model import generate_model
import numpy as np
import os


def experiment2(filename, ret_val):
    mmre = []
    for _ in xrange(30):
        method = guo_random(filename)
        training_data, testing_data = method.generate_test_data(ret_val)
        assert(len(training_data[0]) == len(training_data[-1])), "Something is wrong"
        assert(len(testing_data[0]) == len(testing_data[-1])), "Something is wrong"
        mmre.append(generate_model(training_data, testing_data))

    print round(np.mean(mmre) * 100, 3), round(np.std(mmre) * 100, 3),


def experiment1(filename, ):
    mmre = []
    for _ in xrange(30):
        method = what(filename)
        training_data, testing_data = method.generate_test_data()
        assert(len(training_data[0]) == len(training_data[-1])), "Something is wrong"
        assert(len(testing_data[0]) == len(testing_data[-1])), "Something is wrong"
        mmre.append(generate_model(training_data, testing_data))

    print round(np.mean(mmre) * 100, 3), round(np.std(mmre) * 100, 3),
    return len(training_data[0])

if __name__ == "__main__":
    files = ["./Data/"+f for f in os.listdir("./Data/") if ".csv" in f]
    for file in files:
        print file,
        ret_val = experiment1(file)
        experiment2(file, ret_val)
        print