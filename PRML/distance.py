'''
@Description: The module defines APIs for distance computation.
'''
import sys
sys.path.append('./PRML/')
import numpy as np
from helper import toArray


# Euclidian distance:
def euclid(pt1, pt2):
    pt1, pt2 = toArray(pt1, pt2)
    squareDist = np.power((pt1-pt2), 2)
    return np.sqrt(np.sum(squareDist))

# Manhattan distance:
def manhattan(pt1, pt2):
    pt1, pt2 = toArray(pt1, pt2)
    dist = np.sum(np.abs(pt1-pt2))
    return dist

# Hamming distance:
# TODO: A Hamming