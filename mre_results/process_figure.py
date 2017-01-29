from __future__ import division
import pickle
import numpy as np
import matplotlib.pyplot as plt

"""This is used to generate data for figure 4"""

def find_number_configs(filename):
    def modify_filename(filename):
        """Changing the filename from abolute to relative paths"""
        return "." + filename
    mod_filename = modify_filename(filename)
    num_lines = sum(1 for _ in open(mod_filename)) - 1 # removing the header
    return num_lines


dict_results = pickle.load(open("merged.p", "r"))
files = dict_results.keys()
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

    print file, dumb_ranks[-1], random_ranks[-1], atri_ranks[-1]

