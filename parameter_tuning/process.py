from __future__ import division
import pickle
import os
import numpy as np
dict_store = {}


def run(filename):
    global dict_store
    ds = pickle.load(open(filename))
    ds_names = ds.keys()
    print ds_names
    for name in ds_names:
        dumb_ds = ds[name][0.4]['dumb']
        median_rank = np.median(dumb_ds['min_rank'])
        median_evals = np.median(dumb_ds['train_set_size'])
        if name not in dict_store.keys(): dict_store[name] = []
        dict_store[name].append([median_rank, median_evals])


name_mapping = {'SS20': './Data/sol-6d-c2-obj2.csv',
                'SS9': './Data/wc-6d-c1-obj1.csv',
                'SS8': './Data/Apache_AllMeasurements.csv',
                'SS21': './Data/wc+rs-3d-c4-obj2.csv',
                'SS5': './Data/lrzip.csv',
                'SS4': './Data/WGet.csv',
                'SS7': './Data/HSMGP_num.csv',
                'SS6': './Data/Dune.csv',
                'SS1': './Data/X264_AllMeasurements.csv',
                'SS3': './Data/SQL_AllMeasurements.csv',
                'SS2': './Data/BDBC_AllMeasurements.csv',
                'SS19': './Data/wc+wc-3d-c4-obj2.csv',
                'SS18': './Data/wc+sol-3d-c4-obj2.csv',
                'SS11': './Data/wc-6d-c1-obj2.csv',
                'SS10': './Data/rs-6d-c3_obj1.csv',
                'SS13': './Data/wc+rs-3d-c4-obj1.csv',
                'SS12': './Data/rs-6d-c3_obj2.csv',
                'SS15': './Data/wc+wc-3d-c4-obj1.csv',
                'SS14': './Data/wc+sol-3d-c4-obj1.csv',
                'SS17': './Data/wc-3d-c4_obj2.csv',
                'SS16': './Data/sol-6d-c2-obj1.csv'}


def draw_fig(result):
    def ret_min_rank(l): return [d[1] for d in l]
    def ret_min_eval(l): return [d[0] for d in l]

    import numpy as np
    import matplotlib.pyplot as plt

    f, ax = plt.subplots(3, 7)
    # plt.subplot(3, 2, 1)
    ax[0][0].plot(ret_min_rank(result[name_mapping['SS1']]), ret_min_eval(result[name_mapping['SS1']]), color='r', marker='o')
    ax[0][1].plot(ret_min_rank(result[name_mapping['SS2']]), ret_min_eval(result[name_mapping['SS2']]), color='r', marker='o')
    ax[0][2].plot(ret_min_rank(result[name_mapping['SS3']]), ret_min_eval(result[name_mapping['SS3']]), color='r', marker='o')
    ax[0][3].plot(ret_min_rank(result[name_mapping['SS4']]), ret_min_eval(result[name_mapping['SS4']]), color='r', marker='o')
    ax[0][4].plot(ret_min_rank(result[name_mapping['SS5']]), ret_min_eval(result[name_mapping['SS5']]), color='r', marker='o')
    ax[0][5].plot(ret_min_rank(result[name_mapping['SS6']]), ret_min_eval(result[name_mapping['SS6']]), color='r', marker='o')
    ax[0][6].plot(ret_min_rank(result[name_mapping['SS7']]), ret_min_eval(result[name_mapping['SS7']]), color='r', marker='o')
    ax[1][0].plot(ret_min_rank(result[name_mapping['SS8']]), ret_min_eval(result[name_mapping['SS8']]), color='r', marker='o')
    ax[1][1].plot(ret_min_rank(result[name_mapping['SS9']]), ret_min_eval(result[name_mapping['SS9']]), color='r', marker='o')
    ax[1][2].plot(ret_min_rank(result[name_mapping['SS10']]), ret_min_eval(result[name_mapping['SS10']]), color='r', marker='o')
    ax[1][3].plot(ret_min_rank(result[name_mapping['SS11']]), ret_min_eval(result[name_mapping['SS11']]), color='r', marker='o')
    ax[1][4].plot(ret_min_rank(result[name_mapping['SS12']]), ret_min_eval(result[name_mapping['SS12']]), color='r', marker='o')
    ax[1][5].plot(ret_min_rank(result[name_mapping['SS13']]), ret_min_eval(result[name_mapping['SS13']]), color='r', marker='o')
    ax[1][6].plot(ret_min_rank(result[name_mapping['SS14']]), ret_min_eval(result[name_mapping['SS14']]), color='r', marker='o')
    ax[2][0].plot(ret_min_rank(result[name_mapping['SS15']]), ret_min_eval(result[name_mapping['SS15']]), color='r', marker='o')
    ax[2][1].plot(ret_min_rank(result[name_mapping['SS16']]), ret_min_eval(result[name_mapping['SS16']]), color='r', marker='o')
    ax[2][2].plot(ret_min_rank(result[name_mapping['SS17']]), ret_min_eval(result[name_mapping['SS17']]), color='r', marker='o')
    ax[2][3].plot(ret_min_rank(result[name_mapping['SS18']]), ret_min_eval(result[name_mapping['SS18']]), color='r', marker='o')
    ax[2][4].plot(ret_min_rank(result[name_mapping['SS19']]), ret_min_eval(result[name_mapping['SS19']]), color='r', marker='o')
    ax[2][5].plot(ret_min_rank(result[name_mapping['SS20']]), ret_min_eval(result[name_mapping['SS20']]), color='r', marker='o')
    ax[2][6].plot(ret_min_rank(result[name_mapping['SS21']]), ret_min_eval(result[name_mapping['SS21']]), color='r', marker='o')

    count = 1
    for i in xrange(3):
        for j in xrange(7):
            for cnt, txt in enumerate([1, 2, 3,4 ,5 , 10]):
                ax[i][j].annotate(str(txt), (ret_min_rank(result[name_mapping['SS' + str(count)]])[cnt], ret_min_eval(result[name_mapping['SS' + str(count)]])[cnt]), fontsize=10)
            ax[i][j].set_title('SS' + str(count), fontsize=10)


            labels = ax[i][j].get_xticklabels()
            plt.setp(labels, rotation=90, fontsize=8)

            labels = ax[i][j].get_yticklabels()
            plt.setp(labels, fontsize=8)

            max_val = max(ret_min_eval(result[name_mapping['SS' + str(count)]])) * 1.1
            if max_val == 0: max_val = 1
            min_val = -1

            ax[i][j].set_ylim(min_val, max_val)

            count += 1

    # plt.ylim(-1, 7)
    f.set_size_inches(14,10)
    f.tight_layout()
    f.text(0.005, 0.5, 'Median Minimum Rank Achieved', va='center', rotation='vertical', fontsize=15)
    f.text(0.40, 0.01, 'Number of Measurements (# Counts)', va='center',  fontsize=15)
    plt.savefig('ParameterTuning.png')
    # plt.show()

if __name__ == "__main__":
    from natsort import natsorted
    # name_dict = {}
    # sort_data = sorted(data, key=lambda x:x[-1])
    # for i, d in enumerate(sort_data):
    #     name_dict['SS' + str(i+1)] = d[0]
    # print name_dict
    files = [f for f in os.listdir(".") if ".py" not in f]
    files = ['mre_results_rank_lives1.p', 'mre_results_rank_lives2.p', 'mre_results_rank_lives3.p', 'mre_results_rank_lives4.p', 'mre_results_rank_lives5.p', 'mre_results_rank_lives10.p',]
    for file in files:
        print
        print file
        run(file)

    keys = natsorted(name_mapping.keys())
    for key in keys:
        print key, dict_store[name_mapping[key]]

    draw_fig(dict_store)


    # import pdb
    # pdb.set_trace()
