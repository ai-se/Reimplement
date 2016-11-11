from __future__ import division
import pickle
import os

def policy1(scores, lives=3):
    """
    no improvement in last 3 runs
    """
    temp_lives = lives
    last = scores[0]
    for i,score in enumerate(scores):
        if i > 0:
            if temp_lives == 0:
                return i
            elif score >= last:
                temp_lives -= 1
                last = score
            else:
                temp_lives = lives
                last = score
    return -1

def policy2(scores, lives=3):
    """
    no improvement in last 3 runs
    """
    temp_lives = lives
    last = scores[0]
    for i,score in enumerate(scores):
        if i > 0:
            if temp_lives == 0:
                return i
            elif score >= last:
                temp_lives -= 1
                last = score
            else:
                last = score
    return -1

files = [f for f in os.listdir(".") if ".py" not in f]
for file in files:
    content = pickle.load(open(file, 'rb'))
    breakpoint = policy1(content)
    if breakpoint != -1:
        print "Policy1: ", file, breakpoint, content[breakpoint-4: breakpoint]
    else:
        print "Policy1 didn't work for ", file

    print "- " * 20
    breakpoint = policy2(content)
    if breakpoint != -1:
        print "Policy2: ", file, breakpoint, content[breakpoint - 4: breakpoint]
    else:
        print "Policy2 didn't work for ", file
    print "= " * 40