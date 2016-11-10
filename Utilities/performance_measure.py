from __future__ import division
import numpy as np


def mmre(predicted, actual):
        mre = []
        for i, j in zip(predicted, actual):
            # print ">>  ", i, j, abs(i-j)/abs(j)
            if j==0:
                # print ">>  ", i, j
                continue
            mre.append(abs(i-j)/abs(j))
        return np.mean(mre)