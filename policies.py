from __future__ import division

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