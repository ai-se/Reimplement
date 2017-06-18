from sk import rdivDemo
import pickle

"""
    This program is used to run Skott-Knott Test
"""

dict_result = pickle.load(open("merged.p", "r"))
files = sorted(dict_result.keys())

for file in files:
    # if file not in temp_list: continue
    results = dict_result[file].keys()
    # only train_set
    train_set_keys = [r for r in results if "train_size" in r]
    rank_keys = [r for r in results if "min_rank" in r]
    lists = []
    for rank_key in rank_keys:
        lists.append([rank_key] + dict_result[file][rank_key])
    rdivDemo(file.replace("./Data/", ""), lists, globalMinMax=False, isLatex=True)
