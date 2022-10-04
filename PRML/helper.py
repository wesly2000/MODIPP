'''
@Description: This module includes mainly some farraginous functions.
'''

import numpy as np
from numpy import random
from pandas import DataFrame

# Transmit list- or tuple-like data into a numpy array
# Note that it checks the dimemsions' equality of two points.
def toArray(pt1, pt2):
    assert len(pt1) == len(pt2), 'The dimensions of two points are not equal!'
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    return pt1, pt2

# Extract i-th dimension coordinate from a group of points
# mainly for the purpose of visualization using matplotlib
def dimensionCoord(pts, dim : int):
    assert dim>=0, 'Dimension must be a non-negative integer!'
    coord = []
    for pt in pts:
        coord.append(pt[dim])
    coord = np.array(coord)
    return coord

# Bootstraping a new dataset
def bootstrap(dataset : DataFrame):
    '''
    @dataset: a Dataframe-like dataset
    @return: the sampled dataset of the same size
    '''
    sampleNum = dataset.shape[0]
    index = random.choice(a=sampleNum, size=sampleNum)
    newDataset = dataset.loc[index, :]
    return newDataset

def MoorePenroseInv(matrix):
    tmp =  np.linalg.inv(np.matmul(matrix.T, matrix))
    invMatrix = np.matmul(tmp, matrix.T)
    return invMatrix