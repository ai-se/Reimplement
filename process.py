from __future__ import division
import pickle


def process1(filename):
    dict_content = {}
    content = open(filename).readlines()
    for line in content:
        if "./Data/" in line:
            dataset_name = line.strip().split("/")[-1].split(".")[0]
            dict_content[dataset_name] = {}
            dict_content[dataset_name]['rank_train_set'] = []
            dict_content[dataset_name]['rank_min_rank'] = []
            dict_content[dataset_name]['mmre_train_set'] = []
            dict_content[dataset_name]['mmre_min_rank'] = []
        elif "+" in line:
            temp_content = map(float, line.strip().split(" ")[2:])
            dict_content[dataset_name]['rank_train_set'].append(temp_content[0])
            dict_content[dataset_name]['rank_min_rank'].append(temp_content[1])
            dict_content[dataset_name]['mmre_train_set'].append(temp_content[2])
            dict_content[dataset_name]['mmre_min_rank'].append(temp_content[3])
        else:
            continue

    pickle.dump(dict_content, open('first2.p', 'w'))


def merge_pickle(pickle1, pickle2):
    import pickle
    dict1 = pickle.load(open(pickle1, "r"))
    dict2 = pickle.load(open(pickle2, "r"))
    final = {}
    for key in dict1.keys():
        final[key] = {}
        d1 = dict1[key]
        for d1key in d1.keys():
            final[key][d1key] = d1[d1key]
        d2 = dict2[key+".csv"]
        final[key]['atri_train_set'] = d2['atri_train_set']
        final[key]['atri_min_rank'] = d2['atri_min_rank']
    import pickle
    pickle.dump(final, open("final.p", "w"))

if __name__ == "__main__":
    # process1("results.txt")
    merge_pickle("first2.p", "atri_result.p")