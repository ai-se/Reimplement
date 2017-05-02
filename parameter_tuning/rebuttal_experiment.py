from __future__ import division
import pickle
import os
import numpy as np
dict_store = {}
import matplotlib.pyplot as plt


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


def draw_bar_chart_rank_diff(result):
    global dict_store
    xaxis = []
    yaxis = []
    names = ['SS'+str(i+1) for i in xrange(len(dict_store.keys())-1)]
    all_rank_diff = {
                        '10': [0.0, 0.0, 14.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        '3': [0.5, 0.0, 15.0, 0.5, 0.0, 2.5, 3.5, 0.0, 1.0, 17.5, 1.0, 2.0, 0.0, 0.0, 0.0, 11.5, 0.5, 0.0, 0.0, 1.0, 0.0],
                        '2': [1.0, 4.0, 4.5, 2.5, 0.0, 4.5, 5.0, 0.0, 2.5, 6.0, 0.5, 1.5, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.5, 0.0],
                        '5': [0.0, 0.0, 9.0, 0.0, 0.0, 3.5, 0.0, 0.0, 1.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        '4': [0.0, 1.0, 9.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                     }

    N = len(names)

    space = 9
    ind = np.arange(space, space * (len(names) + 1), space)  # the x locations for the groups
    width = 1.5  # the width of the bars

    fig, ax = plt.subplots()
    lives2 = all_rank_diff['2']
    rects1 = ax.bar(ind, lives2, width, color='#4B514C',label='Lives=2')

    lives3 = all_rank_diff['3']
    rects2 = ax.bar(ind + 1 * width, lives3, width, color='#DD451F',label='Lives=3')

    lives4 = all_rank_diff['4']
    rects3 = ax.bar(ind + 2 * width, lives4, width, color='#626B64',label='Lives=4')

    lives5 = all_rank_diff['5']
    rects4 = ax.bar(ind + 3 * width, lives5, width, color='#79847C',label='Lives=5')

    lives10 = all_rank_diff['10']
    rects5 = ax.bar(ind + 4 * width, lives10, width, color='#919E93',label='Lives=10')

    # add some text for labels, title and axes ticks
    # For shading
    # plt.axvspan(5, 34, color='g', alpha=0.2, lw=0)
    # plt.axvspan(34, 90, color='y', alpha=0.2, lw=0)
    # plt.axvspan(90, 154, color='r', alpha=0.2, lw=0)

    # ax.add_patch ()

    ax.set_ylabel('Rank Difference')
    # ax.set_title('Scores by group and gender')

    ax.set_xticks(ind + 3 * width / 2)
    ax.set_xticklabels(['SS' + str(x + 1) for x in xrange(len(names))], rotation='vertical')

    # ax.set_xlim(3, 157)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fancybox=True, frameon=False)

    # ax.legend((rects1[0], rects2[0], rects3[0]), ('Rank based Approach', 'Progressive Sampling', 'Projective Sampling'))


    # plt.show()

    # fig.set_size_inches(14, 5)
    plt.show()
    # plt.savefig('figure5.png', bbox_inches='tight')


def draw_bar_chart_evals(result):
    global dict_store
    xaxis = []
    yaxis = []
    names = ['SS'+str(i+1) for i in xrange(len(dict_store.keys())-1)]
    all_evals = {
                    '10': [459.0, 1023.0, 1860.0, 74.0, 171.0, 920.0, 1381.0, 75.0, 1151.0, 1535.0, 1151.0, 1535.0, 77.0, 77.0, 77.0, 1145.0, 301.0, 77.0, 77.0, 1143.0, 77.0],
                    '3': [33.0, 39.5, 20.0, 26.5, 37.5, 29.0, 27.0, 23.5, 30.5, 27.0, 28.0, 34.5, 18.0, 23.0, 26.0, 36.5, 23.0, 19.5, 24.0, 28.5, 20.5],
                    '2': [17.0, 16.0, 15.0, 15.5, 19.0, 17.5, 17.0, 16.0, 16.5, 16.0, 15.5, 15.0, 16.0, 14.0, 14.5, 18.5, 17.0, 16.0, 15.0, 16.0, 16.0],
                    '5': [411.0, 447.0, 129.5, 74.0, 171.0, 170.0, 255.5, 75.0, 164.5, 268.5, 448.0, 408.0, 77.0, 62.5, 77.0, 557.0, 133.0, 49.0, 45.0, 331.5, 61.0],
                    '4': [75.5, 129.5, 67.5, 58.5, 78.5, 84.0, 106.5, 56.0, 82.0, 77.0, 113.5, 85.0, 54.5, 32.0, 44.5, 70.5, 49.5, 36.5, 45.0, 97.5, 34.0]
                 }

    N = len(names)

    space = 9
    ind = np.arange(space, space * (len(names) + 1), space)  # the x locations for the groups
    width = 1.5  # the width of the bars

    fig, ax = plt.subplots()
    lives2 = all_evals['2']
    rects1 = ax.bar(ind, lives2, width, color='#4B514C',label='Lives=2', log=True)

    lives3 = all_evals['3']
    rects2 = ax.bar(ind + 1 * width, lives3, width, color='#DD451F',label='Lives=3', log=True)

    lives4 = all_evals['4']
    rects3 = ax.bar(ind + 2 * width, lives4, width, color='#626B64',label='Lives=4', log=True)

    lives5 = all_evals['5']
    rects4 = ax.bar(ind + 3 * width, lives5, width, color='#79847C',label='Lives=5', log=True)

    lives10 = all_evals['10']
    rects5 = ax.bar(ind + 4 * width, lives10, width, color='#919E93',label='Lives=10', log=True)

    # add some text for labels, title and axes ticks
    # For shading
    # plt.axvspan(5, 34, color='g', alpha=0.2, lw=0)
    # plt.axvspan(34, 90, color='y', alpha=0.2, lw=0)
    # plt.axvspan(90, 154, color='r', alpha=0.2, lw=0)

    # ax.add_patch ()

    ax.set_ylabel('Evaluations')
    # ax.set_title('Scores by group and gender')

    ax.set_xticks(ind + 3 * width / 2)
    ax.set_xticklabels(['SS' + str(x + 1) for x in xrange(len(names))], rotation='vertical')

    # ax.set_xlim(3, 157)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fancybox=True, frameon=False)

    # ax.legend((rects1[0], rects2[0], rects3[0]), ('Rank based Approach', 'Progressive Sampling', 'Projective Sampling'))


    # plt.show()

    # fig.set_size_inches(14, 5)
    plt.show()
    # plt.savefig('figure5.png', bbox_inches='tight')


def draw_bar_chart(result):
    global dict_store
    xaxis = []
    yaxis = []
    names = ['SS'+str(i+1) for i in xrange(len(dict_store.keys())-1)]
    all_rank_diff = {
                        '10': [0.0, 0.0, 14.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        '3': [0.5, 0.0, 15.0, 0.5, 0.0, 2.5, 3.5, 0.0, 1.0, 4, 1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.5, 0.0, 0.0, 1.0, 0.0],
                        '2': [1.0, 4.0, 4.5, 2.5, 0.0, 4.5, 5.0, 0.0, 2.5, 6.0, 0.5, 1.5, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.5, 0.0],
                        '5': [0.0, 0.0, 9.0, 0.0, 0.0, 3.5, 0.0, 0.0, 1.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        '4': [0.0, 1.0, 9.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                     }


    all_evals = {
                    '10': [459.0, 1023.0, 1860.0, 74.0, 171.0, 920.0, 1381.0, 75.0, 1151.0, 1535.0, 1151.0, 1535.0, 77.0, 77.0, 77.0, 1145.0, 301.0, 77.0, 77.0, 1143.0, 77.0],
                    '3': [33.0, 39.5, 20.0, 26.5, 37.5, 29.0, 27.0, 23.5, 30.5, 27.0, 28.0, 34.5, 18.0, 23.0, 26.0, 36.5, 23.0, 19.5, 24.0, 28.5, 20.5],
                    '2': [17.0, 16.0, 15.0, 15.5, 19.0, 17.5, 17.0, 16.0, 16.5, 16.0, 15.5, 15.0, 16.0, 14.0, 14.5, 18.5, 17.0, 16.0, 15.0, 16.0, 16.0],
                    '5': [411.0, 447.0, 129.5, 74.0, 171.0, 170.0, 255.5, 75.0, 164.5, 268.5, 448.0, 408.0, 77.0, 62.5, 77.0, 557.0, 133.0, 49.0, 45.0, 331.5, 61.0],
                    '4': [75.5, 129.5, 67.5, 58.5, 78.5, 84.0, 106.5, 56.0, 82.0, 77.0, 113.5, 85.0, 54.5, 32.0, 44.5, 70.5, 49.5, 36.5, 45.0, 97.5, 34.0]
                 }

    import pdb
    pdb.set_trace()
    N = len(names)

    space = 9
    ind = np.arange(space, space * (len(names) + 1), space)  # the x locations for the groups
    width = 1.5  # the width of the bars

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    lives2 = all_rank_diff['2']
    rects1 = ax1.bar(ind, lives2, width, color='#4B514C',label='Lives=2')

    lives3 = all_rank_diff['3']
    rects2 = ax1.bar(ind + 1 * width, lives3, width, color='#DD451F',label='Lives=3')

    lives4 = all_rank_diff['4']
    rects3 = ax1.bar(ind + 2 * width, lives4, width, color='#626B64',label='Lives=4')

    lives5 = all_rank_diff['5']
    rects4 = ax1.bar(ind + 3 * width, lives5, width, color='#79847C',label='Lives=5')

    lives10 = all_rank_diff['10']
    rects5 = ax1.bar(ind + 4 * width, lives10, width, color='#919E93',label='Lives=10')

    ax1.set_ylabel('Rank Difference')
    # ax.set_title('Scores by group and gender')

    # ax1.set_xticks(ind + 3 * width / 2)
    # ax1.set_xticklabels(['SS' + str(x + 1) for x in xrange(len(names))], rotation='vertical')

    lives2 = all_evals['2']
    rects1 = ax2.bar(ind, lives2, width, color='#4B514C', label='Lives=2', log=True)

    lives3 = all_evals['3']
    rects2 = ax2.bar(ind + 1 * width, lives3, width, color='#DD451F', label='Lives=3', log=True)

    lives4 = all_evals['4']
    rects3 = ax2.bar(ind + 2 * width, lives4, width, color='#626B64', label='Lives=4', log=True)

    lives5 = all_evals['5']
    rects4 = ax2.bar(ind + 3 * width, lives5, width, color='#79847C', label='Lives=5', log=True)

    lives10 = all_evals['10']
    rects5 = ax2.bar(ind + 4 * width, lives10, width, color='#919E93', label='Lives=10', log=True)
    # ax.set_xlim(3, 157)

    ax2.set_ylabel('Evaluations')
    # ax.set_title('Scores by group and gender')

    ax2.set_xticks(ind + 3 * width / 2)
    ax2.set_xticklabels(['SS' + str(x + 1) for x in xrange(len(names))], rotation='vertical')

    # ax.set_xlim(3, 157)

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=5, fancybox=True, frameon=False)

    # ax.legend((rects1[0], rects2[0], rects3[0]), ('Rank based Approach', 'Progressive Sampling', 'Projective Sampling'))


    # plt.show()

    fig.set_size_inches(14, 5)
    # plt.show()
    plt.savefig('rebuttal_merge.png', bbox_inches='tight')


if __name__ == "__main__":
    global dict_store
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

    # draw_bar_chart_rank_diff(dict_store)
    # draw_bar_chart_evals(dict_store)
    draw_bar_chart(dict_store)
    # draw_fig(dict_store)


    # import pdb
    # pdb.set_trace()
