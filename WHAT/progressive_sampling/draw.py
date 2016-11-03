from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

content = pickle.load( open( "./data/save.p", "rb" ) )
files = content.keys()
for file in files:
    measures = sorted(content[file].keys())
    al_mmres = []
    random_mmres = []
    for measure in measures:
        al_mmre = content[file][measure]['al_mmre']
        random_mmre = content[file][measure]['random_mmre']
        al_mmres.append(np.mean(al_mmre))
        random_mmres.append(np.mean(random_mmre))
    plt.plot(measures, al_mmres, color='b', label='ActiveLearning')
    plt.plot(measures, random_mmres, color='r', label='RandomS')
    plt.xlabel('measures')
    plt.ylabel('mmre')
    plt.title(file.split('/')[-1])
    plt.legend(loc='best')


    plt.savefig("./figures/" + file.split('/')[-1].split('.')[0])