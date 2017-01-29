import matplotlib.pyplot as plt

data = [
    # datasetname, dumbmre, dumbstd, dumbevals, randommre, randomstd, randomevals, atrimre, atristd, atrievals, accuracy from figure1 of the paper
    ["./Data/Apache_AllMeasurements.csv", 10.7923400483, 4.30570224034, 29.5, 10.4947292635, 2.20430123136, 16.0, 7.54820448973, 1.02106706701, 49.0, 7.17],
    ["./Data/BDBC_AllMeasurements.csv", 110.479335881, 187.802451244, 38.0, 11.8572934856, 4.30291195193, 42.5, 1.40528125434, 0.835513510745, 186.0, 0.48],
    ["./Data/Dune.csv", 14.9476364347, 2.94470453023, 36.0, 10.4241382255, 0.852644275258, 99.0, 10.7840541533, 1.06376053277, 117.0, 6.25],
    ["./Data/HSMGP_num.csv", 28.4925806894, 6.3190261939, 26.5, 10.2280499171, 0.462206395751, 444.0, 10.877402825, 0.742661331947, 383.0, 6.88],
    ["./Data/SQL_AllMeasurements.csv", 5.86748929191, 0.484176841033, 22.0, 6.13162555204, 0.841248582473, 10.0, 5.47668186009, 0.326518039763, 66.0, 4.41],
    ["./Data/WGet.csv", 9.67864308777, 11.9126425088, 27.0, 11.2691693847, 10.6807762614, 16.5, 12.1150493787, 12.0269102756, 23.0, 4.71],
    ["./Data/X264_AllMeasurements.csv", 6.43736284941, 4.22374067851, 31.5, 9.8685240802, 1.42535845389, 12.5, 0.751253662102, 0.187911591044, 151.0, 0.19],
    ["./Data/lrzip.csv", 95.3716768331, 184.552495732, 28.0, 12.9970921642, 4.34355122863, 87.5, 24.5860492819, 9.95373861074, 49.0, 6.07],
    ["./Data/rs-6d-c3_obj1.csv", 421.042970956, 522.879249452, 32.0, 13.7337428323, 8.5542189313, 599.5, 12.1311203278, 3.82724520753, 678.0, 8.4],
    ["./Data/rs-6d-c3_obj2.csv", 3650.66714862, 8035.48329532, 27.5, 13.5995547003, 5.83251455828, 687.5, 11.7792618959, 5.9591126095, 835.0, 9.01],
    ["./Data/sol-6d-c2-obj1.csv", 643.593480935, 1743.01883524, 37.0, 43.6706927286, 22.377177188, 577.0, 59.6853689508, 32.0536509917, 188.0, 38.08],
    ["./Data/sol-6d-c2-obj2.csv", 2653.46711207, 3944.90370138, 33.5, 80.6249317966, 23.9129599203, 576.0, 153.869243982, 107.110325652, 251.0, 76.52],
    # ["./Data/sort_256_obj2.csv", 11.5712032183, 3.87771922132, 25.5, 10.0586679798, 2.44405242568, 26.0, 20.862548876, 3.74071290731, 4.0, 6.79],
    ["./Data/wc+rs-3d-c4-obj1.csv", 34.8070233671, 28.5367725653, 23.5, 17.6727526682, 6.0489703848, 43.0, 19.3442173065, 4.76740956441, 28.0, 10.46],
    ["./Data/wc+rs-3d-c4-obj2.csv", 4219.83055425, 17355.9942052, 19.0, 105.583909043, 129.957739811, 43.0, 232.074858955, 518.975346651, 45.0, 76.55],
    ["./Data/wc+sol-3d-c4-obj1.csv", 37.1469049896, 25.4539940247, 26.0, 20.8223753716, 12.4935049708, 43.0, 27.8280265354, 15.9387424543, 28.0, 12.36],
    ["./Data/wc+sol-3d-c4-obj2.csv", 3731.3645478, 8159.94720512, 21.0, 138.098828117, 160.052893102, 43.0, 50.5793154978, 62.3850186383, 56.0, 46.08],
    ["./Data/wc+wc-3d-c4-obj1.csv", 31.6497529061, 30.7841955488, 26.0, 18.7216467377, 7.13603475529, 43.0, 26.8501276869, 11.9082288625, 27.0, 15.9],
    ["./Data/wc+wc-3d-c4-obj2.csv", 3159.59661431, 11636.1618333, 23.0, 93.9384111155, 91.27189707, 43.0, 90, 0.0, 48, 48.75],
    ["./Data/wc-3d-c4_obj2.csv", 12402.4561374, 31085.8027157, 28.5, 52.5777728899, 42.782323966, 155.0, 895.991096589, 2555.03744773, 59.0, 44.45],
    ["./Data/wc-6d-c1-obj1.csv", 141.305549567, 117.172292763, 26.5, 10.8449550879, 1.44458681925, 388.5, 13.1048972691, 3.33600531343, 361.0, 7.44],
    ["./Data/wc-6d-c1-obj2.csv", 690.812872709, 2068.20619916, 28.0, 10.4653064851, 0.63580925096, 336.0, 20.5970699707, 7.97878382382, 125.0, 8.47],

]

import numpy as np
import matplotlib.pyplot as plt
data = sorted(data, key=lambda x: x[-1])
N = len(data)
dumb_evals = [d[3] for d in data]

space = 7
ind = np.arange(space, space*(len(data)+1), space)  # the x locations for the groups
width = 1.5        # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, dumb_evals, width, color='#f0f0f0', log=True, label='Rank based Approach')

random_evals = [d[6] for d in data]
rects2 = ax.bar(ind + 1 * width, random_evals, width, color='#bdbdbd', log=True, label='Progessive Sampling')

atri_evals = [d[9] for d in data]
rects3 = ax.bar(ind + 2 * width, atri_evals, width, color='#636363', log=True, label='Projective Sampling')

# add some text for labels, title and axes ticks
# For shading
# plt.axvspan(5, 34, color='g', alpha=0.2, lw=0)
# plt.axvspan(34, 90, color='y', alpha=0.2, lw=0)
# plt.axvspan(90, 154, color='r', alpha=0.2, lw=0)

# ax.add_patch ()

ax.set_ylabel('Number of Evaluations')
# ax.set_title('Scores by group and gender')

ax.set_xticks(ind + 3*width / 2)
ax.set_xticklabels(['SS'+str(x+1) for x in xrange(len(data))], rotation='vertical')

ax.set_xlim(3, 157)
ax.set_ylim(1, 1000)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True, frameon=False)

# ax.legend((rects1[0], rects2[0], rects3[0]), ('Rank based Approach', 'Progressive Sampling', 'Projective Sampling'))


# plt.show()

fig.set_size_inches(14, 5)
# plt.show()
plt.savefig('figure6.png', bbox_inches='tight')
