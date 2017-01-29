from __future__ import division
import pickle
import numpy as np
import matplotlib.pyplot as plt

"""This is used to generate data for figure 4"""

# Accuracy scores from figure 1
acc_data = {}
acc_data["./Data/Apache_AllMeasurements.csv"] = 7.17
acc_data["./Data/BDBC_AllMeasurements.csv"] = 0.48
acc_data["./Data/Dune.csv"] = 6.25
acc_data["./Data/HSMGP_num.csv"] = 6.88
acc_data["./Data/lrzip.csv"] = 6.07
acc_data["./Data/rs-6d-c3_obj1.csv"] = 8.4
acc_data["./Data/rs-6d-c3_obj2.csv"] = 9.01
acc_data["./Data/sol-6d-c2-obj1.csv"] = 38.08
acc_data["./Data/sol-6d-c2-obj2.csv"] = 76.52
acc_data["./Data/sort_256_obj2.csv"] = 6.79
acc_data["./Data/SQL_AllMeasurements.csv"] = 4.41
acc_data["./Data/wc-3d-c4_obj2.csv"] = 44.45
acc_data["./Data/wc-6d-c1-obj1.csv"] = 7.44
acc_data["./Data/wc-6d-c1-obj2.csv"] = 8.47
acc_data["./Data/wc+rs-3d-c4-obj1.csv"] = 10.46
acc_data["./Data/wc+rs-3d-c4-obj2.csv"] = 76.55
acc_data["./Data/wc+sol-3d-c4-obj1.csv"] = 12.36
acc_data["./Data/wc+sol-3d-c4-obj2.csv"] = 46.08
acc_data["./Data/wc+wc-3d-c4-obj1.csv"] = 15.9
acc_data["./Data/wc+wc-3d-c4-obj2.csv"] = 48.75
acc_data["./Data/WGet.csv"] = 4.71
acc_data["./Data/X264_AllMeasurements.csv"] = 0.19

def find_number_configs(filename):
    def modify_filename(filename):
        """Changing the filename from abolute to relative paths"""
        return "." + filename
    mod_filename = modify_filename(filename)
    num_lines = sum(1 for _ in open(mod_filename)) - 1 # removing the header
    return num_lines


dict_results = pickle.load(open("merged.p", "r"))
files = sorted(dict_results.keys())
for file in files:
    filename = file.split("/")[-1].split(".")[0]
    dumb_mres = []
    dumb_stds = []
    dumb_evals = []
    dumb_ranks = []


    random_mres = []
    random_stds = []
    random_evals = []
    random_ranks = []

    atri_mres = []
    atri_stds = []
    atri_evals = []
    atri_ranks = []

    fractions = sorted(dict_results[file].keys())
    for fraction in [0.4]:# fractions:
        algorithms = dict_results[file][fraction].keys()
        dumb_mres.append(np.mean(dict_results[file][fraction]['dumb']['mres'])*100)
        dumb_stds.append(np.std(dict_results[file][fraction]['dumb']['mres'])*100)

        random_mres.append(np.mean(dict_results[file][fraction]['random-progressive']['mres'])*100)
        random_stds.append(np.std(dict_results[file][fraction]['random-progressive']['mres'])*100)

        atri_mres.append(np.mean(dict_results[file][fraction]['random-projective']['mres'])*100)
        atri_stds.append(np.std(dict_results[file][fraction]['random-projective']['mres'])*100)

        dumb_evals.append(np.median(dict_results[file][fraction]['dumb']['train_set_size']))
        random_evals.append(np.median(dict_results[file][fraction]['random-progressive']['train_set_size']))
        atri_evals.append(np.median(dict_results[file][fraction]['random-projective']['train_set_size']))

        dumb_ranks.append(np.median(dict_results[file][fraction]['dumb']['min_rank']))
        random_ranks.append(np.median(dict_results[file][fraction]['random-progressive']['min_rank']))
        atri_ranks.append(np.median(dict_results[file][fraction]['random-projective']['min_rank']))

    print "[\"" + file + "\", " + str(dumb_mres[-1]) + ", " + str(dumb_stds[-1]) + ", "  + str(dumb_evals[-1]) + ", "\
          + str(random_mres[-1]) + ", " + str(random_stds[-1]) + ", " + str(random_evals[-1]) + ", " \
          + str(atri_mres[-1]) + ", " +str(atri_stds[-1]) + ", " + str(atri_evals[-1]) + ", " \
          + str(acc_data[file]) + "],"

# copy this data to figure5.py

