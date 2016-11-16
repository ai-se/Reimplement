from __future__ import division
import pickle
import numpy as np


dict1 = pickle.load(open("atri_result.p", "r"))
dict2 = pickle.load(open("mre_results_rank.p", 'r'))
dict3 = {}
name_replace = {
        "HSMGP_num.csv":"HSMGP.csv",
        "BDBC_AllMeasurements.csv":"BDBC.csv",
        "SQL_AllMeasurements.csv":"SQL.csv",
        "Apache_AllMeasurements.csv":"Apache.csv"

    }
files = dict2.keys()
for file in files:
    dict3[file] = {}
    try:
        f1 = dict1[file.split("/")[-1]]
    except:
        f1 = dict1[name_replace[file.split("/")[-1]]]
    f2 = dict2[file]
    fractions = f2.keys()
    for fraction in fractions:
        dict3[file][fraction] = {}
        ff1 = f1[round(fraction, 1)]


        ff2 = f2[fraction]
        dict3[file][fraction] = ff2
        dict3[file][fraction]['random-projective'] = {}
        if len(ff1.keys()) != 0:
            dict3[file][fraction]['random-projective']['mres'] = ff1['atri_min_mre']
            dict3[file][fraction]['random-projective']['train_set_size'] = ff1['atri_train_set']
            dict3[file][fraction]['random-projective']['min_rank'] = ff1['atri_min_rank']
        else:
            dict3[file][fraction]['random-projective']['mres'] = [-1 for _ in xrange(20)]
            dict3[file][fraction]['random-projective']['train_set_size'] = [-1 for _ in xrange(20)]
            dict3[file][fraction]['random-projective']['min_rank'] = [-1 for _ in xrange(20)]

print len(dict3)
pickle.dump(dict3, open('merged.p', 'w'))