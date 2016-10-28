from __future__ import division
import numpy as np


def mmre(predicted, actual):
        mre = []
        for i, j in zip(predicted, actual):
            mre.append(abs(i-j)/abs(j))
        return np.mean(mre)