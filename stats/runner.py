from sk import rdivDemo
import pickle

names = {
    "atri_train_set": "sarkar-projective",
    "rank_train_set": "dumb",
    "mmre_train_set": "random-progressive",
    'rank_min_rank' : "dumb",
    'atri_min_rank' : "sarkar-projective",
    'mmre_min_rank' : "random-progressive",
}

data_sizes = {
    "Apache_AllMeasurements":192,
    "BDBC_AllMeasurements":2561,
    "Dune":2305,
    "HSMGP_num":3457,
    "SQL_AllMeasurements":4654,
    "WGet":189,
    "X264_AllMeasurements":1153,
    "lrzip":433,
    "rs-6d-c3_obj1":3840,
    "rs-6d-c3_obj2":3840,
    "sol-6d-c2-obj1":2866,
    "sol-6d-c2-obj2":2862,
    "sort_256_obj2":206,
    "wc+rs-3d-c4-obj1":196,
    "wc+rs-3d-c4-obj2":196,
    "wc+sol-3d-c4-obj1":196,
    "wc+sol-3d-c4-obj2":196,
    "wc+wc-3d-c4-obj1":196,
    "wc+wc-3d-c4-obj2":196,
    "wc-3d-c4_obj2":756,
    "wc-6d-c1-obj1":2880,
    "wc-6d-c1-obj2":2880,
}

dict_result = pickle.load(open("final.p", "r"))
files = dict_result.keys()

for file in files:
    results = dict_result[file].keys()
    # only train_set
    train_set_keys = [r for r in results if "train_set" in r]
    rank_keys = [r for r in results if "min_rank" in r]
    lists = []
    print "##", file, "(", data_sizes[file], ")"
    print "### Training Set Size"
    print "```"
    for train_set_key in train_set_keys:
        lists.append([names[train_set_key]] + dict_result[file][train_set_key])
    rdivDemo(lists, lessismore=True)
    print "```"
    print " ### Minimum Rank Found"
    print "```"
    lists = []
    for rank_key in rank_keys:
        lists.append([names[rank_key]] + dict_result[file][rank_key])
    rdivDemo(lists, globalMinMax=False, lessismore=True)
    print "```"
