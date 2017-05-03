import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
# name_mapping = {'SS20': './Data/sol-6d-c2-obj2.csv',
#                 'SS9': './Data/wc-6d-c1-obj1.csv',
#                 'SS8': './Data/Apache_AllMeasurements.csv',
#                 'SS21': './Data/wc+rs-3d-c4-obj2.csv',
#                 'SS5': './Data/lrzip.csv',
#                 'SS4': './Data/WGet.csv',
#                 'SS7': './Data/HSMGP_num.csv',
#                 'SS6': './Data/Dune.csv',
#                 'SS1': './Data/X264_AllMeasurements.csv',
#                 'SS3': './Data/SQL_AllMeasurements.csv',
#                 'SS2': './Data/BDBC_AllMeasurements.csv',
#                 'SS19': './Data/wc+wc-3d-c4-obj2.csv',
#                 'SS18': './Data/wc+sol-3d-c4-obj2.csv',
#                 'SS11': './Data/wc-6d-c1-obj2.csv',
#                 'SS10': './Data/rs-6d-c3_obj1.csv',
#                 'SS13': './Data/wc+rs-3d-c4-obj1.csv',
#                 'SS12': './Data/rs-6d-c3_obj2.csv',
#                 'SS15': './Data/wc+wc-3d-c4-obj1.csv',
#                 'SS14': './Data/wc+sol-3d-c4-obj1.csv',
#                 'SS17': './Data/wc-3d-c4_obj2.csv',
#                 'SS16': './Data/sol-6d-c2-obj1.csv'}
#
# inv_name_mapping = {v: k for k, v in name_mapping.iteritems()}
# another = {'SS19': 'rebuttal_rank_diffwc+wc-3d-c4-obj2.p', 'SS9': 'rebuttal_rank_diffwc-6d-c1-obj1.p', 'SS8': 'rebuttal_rank_diffApache_AllMeasurements.p', 'SS18': 'rebuttal_rank_diffwc+sol-3d-c4-obj2.p', 'SS5': 'rebuttal_rank_difflrzip.p', 'SS4': 'rebuttal_rank_diffWGet.p', 'SS7': 'rebuttal_rank_diffHSMGP_num.p', 'SS6': 'rebuttal_rank_diffDune.p', 'SS1': 'rebuttal_rank_diffX264_AllMeasurements.p', 'SS3': 'rebuttal_rank_diffSQL_AllMeasurements.p', 'SS2': 'rebuttal_rank_diffBDBC_AllMeasurements.p', 'SS20': 'rebuttal_rank_diffsol-6d-c2-obj2.p', 'SS21': 'rebuttal_rank_diffwc+rs-3d-c4-obj2.p', 'SS11': 'rebuttal_rank_diffwc-6d-c1-obj2.p', 'SS10': 'rebuttal_rank_diffrs-6d-c3_obj1.p', 'SS13': 'rebuttal_rank_diffwc+rs-3d-c4-obj1.p', 'SS12': 'rebuttal_rank_diffrs-6d-c3_obj2.p', 'SS15': 'rebuttal_rank_diffwc+wc-3d-c4-obj1.p', 'SS14': 'rebuttal_rank_diffwc+sol-3d-c4-obj1.p', 'SS17': 'rebuttal_rank_diffwc-3d-c4_obj2.p', 'SS16': 'rebuttal_rank_diffsol-6d-c2-obj1.p'}
#
#
# files = [ f for f in os.listdir(".") if "rebuttal_rank_diff" in f]
names = ['SS' + str(i+1) for i in xrange(21)]
# data = []
# for name in names:
#     file = another[name]
#     content = pickle.load(open(file, 'rb'))
#     key = content.keys()[-1]
#     data.append(np.mean(content[key]['rank_rd']))

data = [0.0, 0.0, 7.050000000000001, 8.050000000000001, 0.0, 0.0, 0.0, 0.0, 6.050000000000001, 13.050000000000001, 4.050000000000001, 5.050000000000001, 15.050000000000001, 15.050000000000001, 0.20000000000000001, 12.050000000000001, 11.050000000000001, 0.20000000000000001, 5.050000000000001, 8.050000000000001, 4.050000000000001]

gap = 35

left, width = .53, .5
bottom, height = .25, .5
right = left + width
top = bottom + height

f, ax1 = plt.subplots()


print len([gap*(i+1) for i in xrange(13, 21)]), len(data[13:]), len(data)
# for dumb learner
ax1.scatter([gap*(i+1) for i in xrange(0, 4)], data[:4], color='g', marker='v', s=34)
ax1.scatter([gap*(i+1) for i in xrange(4, 13)], data[4:13], color='y', marker='o', s=34)
ax1.scatter([gap*(i+1) for i in xrange(13, 21)], data[13:], color='r', marker='x', s=34)

ax1.tick_params(axis=u'both', which=u'both',length=0)
ax1.set_ylim(-2,16)
ax1.set_xlim(10, 770)
ax1.set_title('Rank-based  - Finding worst performing configuration')
ax1.set_ylabel("Rank Difference (RD)")
plt.sca(ax1)
plt.xticks([gap*(i+1) for i in xrange(0, 21)], names, rotation=90)

from matplotlib.lines import Line2D
circ3 = Line2D([0], [0], linestyle="none", marker="x", alpha=0.3, markersize=10, color="r")
circ1 = Line2D([0], [0], linestyle="none", marker="v", alpha=0.4, markersize=10, color="g")
circ2 = Line2D([0], [0], linestyle="none", marker="o", alpha=0.3, markersize=10, color="y")
plt.figlegend((circ1, circ2, circ3), ('<5%', '5%<MMRE<10%', '>10%'), frameon=False, loc='lower center',
              bbox_to_anchor=(0.5, -0.036),fancybox=True, ncol=3)
f.set_size_inches(8, 5.5)
plt.savefig('rebuttal_worst_performance.png', bbox_inches='tight')