import matplotlib.pyplot as plt

"""
Trying to redraw the figure based on Dr. Apel's suggestion.
"""

old_data = [["./Data/Apache_AllMeasurements.csv",0,0,0,7.17],
        ["./Data/BDBC_AllMeasurements.csv",1,1,0,0.48],
        ["./Data/Dune.csv",4.5,2,4,6.25],
        ["./Data/HSMGP_num.csv",2,0,0,6.88],
        ["./Data/lrzip.csv",0,0,0,6.07],
        ["./Data/rs-6d-c3_obj1.csv",4,0,0,8.4],
        ["./Data/rs-6d-c3_obj2.csv",7.5,1,1,9.01],
        ["./Data/sol-6d-c2-obj1.csv",2,3.5,2.5,38.08],
        ["./Data/sol-6d-c2-obj2.csv",1,0,0,76.52],
        #["./Data/sort_256_obj2.csv",0,0,0,6.79],
        ["./Data/SQL_AllMeasurements.csv",5.5,7.5,12.5,4.41],
        ["./Data/wc-3d-c4_obj2.csv",0,0,0,44.45],
        ["./Data/wc-6d-c1-obj1.csv",0,0,0,7.44],
        ["./Data/wc-6d-c1-obj2.csv",2.5,0,0.5,8.47],
        ["./Data/wc+rs-3d-c4-obj1.csv",0,0,0,10.46],
        ["./Data/wc+rs-3d-c4-obj2.csv",0,0,0,76.55],
        ["./Data/wc+sol-3d-c4-obj1.csv",0,0,0,12.36],
        ["./Data/wc+sol-3d-c4-obj2.csv",0,0,0,46.08],
        ["./Data/wc+wc-3d-c4-obj1.csv",0,0,0,15.9],
        ["./Data/wc+wc-3d-c4-obj2.csv",0,0,1,48.75],
        ["./Data/WGet.csv",1,1,1,4.71],
        ["./Data/X264_AllMeasurements.csv",0,1,0,0.19]]

data = [['SS' + str(i+1)] + d for i,d in enumerate(sorted(old_data, key=lambda k:k[-1]))]
gap = 35

left, width = .53, .5
bottom, height = .25, .5
right = left + width
top = bottom + height

f, ((ax1, ax2, ax3)) = plt.subplots(1, 3)

print ">> ", len([d[2] for d in data if 5 < d [-1] <= 10.5]), len([5*(i+1) for i in xrange(4, 13)])

# for dumb learner
ax1.scatter([gap*(i+1) for i in xrange(0, 4)], [d[2] for d in data if d [-1] <= 5], color='g', marker='v', s=34)
ax1.scatter([gap*(i+1) for i in xrange(4, 13)], [d[2] for d in data if 5 < d [-1] <= 10.5], color='y', marker='o', s=34)
ax1.scatter([gap*(i+1) for i in xrange(13, 21)], [d[2] for d in data if d [-1] > 10.5], color='r', marker='x', s=34)

ax1.tick_params(axis=u'both', which=u'both',length=0)
ax1.set_ylim(-2,14)
ax1.set_xlim(10, 770)
ax1.set_title('Rank-based', fontsize=16)
ax1.set_ylabel("Rank Difference (RD)", fontsize=16)

# ax1.set_xlabel("Accuracy")
# ax1.set_yscale('log')

ax2.set_ylim(-2,14)
ax2.set_xlim(10, 770)
ax2.scatter([gap*(i+1) for i in xrange(0, 4)], [d[3] for d in data if d [-1] <= 5], marker='v', color='g', s=34)
ax2.scatter([gap*(i+1) for i in xrange(4, 13)], [d[3] for d in data if 5 < d [-1] <= 10.5], marker='o', color='y', s=34)
ax2.scatter([gap*(i+1) for i in xrange(13, 21)], [d[3] for d in data if d [-1] > 10.5], marker='x', color='r', s=34)

ax2.tick_params(axis=u'both', which=u'both',length=0)
ax2.set_title('Progressive Sampling', fontsize=16)
ax2.set_ylabel("Rank Difference (RD)", fontsize=16)
# ax2.set_xlabel("Accuracy")

ax3.set_ylim(-2,14)
ax3.set_xlim(10, 770)
ax3.scatter([gap*(i+1) for i in xrange(0, 4)], [d[4] for d in data if d [-1] <= 5], marker='v', color='g', s=34)
ax3.scatter([gap*(i+1) for i in xrange(4, 13)], [d[4] for d in data if 5 < d[-1] <= 10.5], marker='o', color='y', s=34)
ax3.scatter([gap*(i+1) for i in xrange(13, 21)], [d[4] for d in data if d [-1] > 10.5 and d[4]!= -1], marker='x', color='r', s=34)

ax3.tick_params(axis=u'both', which=u'both',length=0)
ax3.set_title('Projective Sampling', fontsize=16)
ax3.set_ylabel("Rank Difference (RD)", fontsize=16)
# ax3.set_xlabel("Accuracy")

from matplotlib.lines import Line2D

circ3 = Line2D([0], [0], linestyle="none", marker="x", alpha=0.3, markersize=10, color="r")
circ1 = Line2D([0], [0], linestyle="none", marker="v", alpha=0.4, markersize=10, color="g")
circ2 = Line2D([0], [0], linestyle="none", marker="o", alpha=0.3, markersize=10, color="y")

plt.sca(ax1)
plt.xticks([gap*(i+1) for i in xrange(0, 21)], [d[0] for d in data], rotation=90, fontsize=12)

plt.sca(ax2)
plt.xticks([gap*(i+1) for i in xrange(0, 21)], [d[0] for d in data], rotation=90, fontsize=12)

plt.sca(ax3)
plt.xticks([gap*(i+1) for i in xrange(0, 21)], [d[0] for d in data], rotation=90, fontsize=12)

plt.figlegend((circ1, circ2, circ3), ('<5%', '5%<MMRE<10%', '>10%'), frameon=False, loc='lower center',
              bbox_to_anchor=(0.4, -0.04),fancybox=True, ncol=3, fontsize=16)

f.set_size_inches(22, 5.5)
# plt.show()
plt.savefig('figure4_2.png', bbox_inches='tight')
