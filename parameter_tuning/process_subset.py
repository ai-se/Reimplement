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

    plt.plot(ret_min_rank(result[name_mapping['SS10']]), ret_min_eval(result[name_mapping['SS10']]), color='r', marker='o')
    plt.xlabel('Number of Measurements (# counts)')
    plt.ylabel('Median Minimum Rank Found')
    plt.title('SS10')
    plt.show()


def average_score(result):
    xaxis = []
    yaxis = []
    for i in xrange(5):
        yaxis.append(np.median([result[key][i][0] for key in result]))
        xaxis.append(np.median([result[key][i][1] for key in result]))

    import matplotlib.pyplot as plt

    plt.plot(xaxis, yaxis, color='r', marker='o')
    plt.annotate('lives='+str(2), (21, 0.75), fontsize=12)
    plt.annotate('lives='+str(3), (32, 0.50), fontsize=12)
    plt.annotate('lives='+str(4), (70, 0.021), fontsize=12)
    plt.annotate('lives='+str(5), (150, 0.021), fontsize=12)
    plt.annotate('lives='+str(10), (350, 0.021), fontsize=12)

    plt.xlabel('Number of Measurements (# counts)')
    plt.ylabel('Median Minimum Rank Found')
    # plt.title('Median Trade-off Curve')
    plt.ylim(-0.1, 1)
    # plt.show()
    plt.savefig('figure8.png')


if __name__ == "__main__":
    from natsort import natsorted
    # name_dict = {}
    # sort_data = sorted(data, key=lambda x:x[-1])
    # for i, d in enumerate(sort_data):
    #     name_dict['SS' + str(i+1)] = d[0]
    # print name_dict
    files = [f for f in os.listdir(".") if ".py" not in f]
    files = [ 'mre_results_rank_lives2.p', 'mre_results_rank_lives3.p', 'mre_results_rank_lives4.p', 'mre_results_rank_lives5.p', 'mre_results_rank_lives10.p',]
    for file in files:
        print
        print file
        run(file)

    keys = natsorted(name_mapping.keys())
    for key in keys:
        print key, dict_store[name_mapping[key]]

    average_score(dict_store)
    # draw_fig(dict_store)


    # import pdb
    # pdb.set_trace()
