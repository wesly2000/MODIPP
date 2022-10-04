'''
@Description: This file implements DBSCAN algorithm
'''

import sys
sys.path.append('./PRML/')

import numpy as np
import distance
from queue import Queue

# Testing:
from matplotlib import pyplot as plt
# from PRML.helper import dimensionCoord as coord

# Find the neighbours of the center within a epsilon-area, there are
# now two area shapes: Euclidian(circle), Manhattan(Square)
# it also returns the number of neighbours
def findNeighbours(pts : set, center, dist, shape='euclid'):
    '''
    @pts: All points to be considered
    @center: the target we consider as the center of @pts.
    @shape: the shape of area centered around @center, euclid for circle,
    manhattan for square.
    @dist: The radius of area centered around @center.
    return: 
        @_neighbour: The neighbours of center, i.e., points that have distance shorter than @dist
        @_len: The number of neighbours
    '''
    _neighbour = set()
    for pt in iter(pts):
        d = 0
        if shape == 'euclid':
            d = distance.euclid(center, pt)
        elif shape == 'manhattan':
            d = distance.manhattan(center, pt)
        
        if d<=dist:
            _neighbour.add(pt)
    _len = len(_neighbour)
    return _neighbour, _len

# The implement of DBSCAN algorithm
def dbscan(pts : set, minPts : int, dist, shape='euclid'):
    '''
    @dataset: The data to be clustered
    @return: The partition of dataset
    '''
    assert dist>0, 'The radius of the neighbourhood must larger than 0!'
    # Step 1: Find the core points:
    cores = set()
    for pt in iter(pts):
        _, neighbourNum = findNeighbours(pts, pt, dist, shape)
        if neighbourNum>=minPts:
            cores.add(pt)
    # Step 2: Clustering
    k = 0
    neverVisited = pts
    clusterFamilies = []
    connected = set()
    coresOld = cores
    coresNum = len(coresOld)
    while coresNum>0:
        for core in iter(cores):
            neverVisitedOld = neverVisited
            # Link contains all density-linked points of the core
            link = Queue()
            link.put(core)
            neverVisited = neverVisited-set(core)
            while link.empty() == False:
                q = link.get()
                neighbours, neighbourNum = findNeighbours(pts, q, dist, shape)
                if neighbourNum<minPts:
                    continue
                else:
                    newVisited = neverVisited&neighbours
                    connected = connected | newVisited
                    for reachable in newVisited:
                        link.put(reachable)
                    # remove the points reachable from neverVisited
                    neverVisited = neverVisited - newVisited
            k += 1
            # Generating a clutering family
            cluteringFamily = neverVisitedOld-neverVisited
            clusterFamilies.append(cluteringFamily)
            # Moreover, since all the density-linked cores belong to the same clustering
            # family, we don't need to check them anymore after a family is partitioned. 
            cores = cores-cluteringFamily
            if cores != coresOld:
                coresOld = cores
                coresNum = len(coresOld)
                break
    return clusterFamilies, neverVisited