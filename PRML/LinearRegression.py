import sys
sys.path.append('./PRML/')
import numpy as np
import cupy as cp
import pandas as pd
from numpy import random
from pandas import DataFrame
import sys
import copy

from kernel import gaussKernel
from helper import *

def simpleDataset(func:str='sin', size:int=25, precision:float=5, low:float=-1, high:float=1):
    '''
        Description: Draw @size points from function @func, and add @perturb noise for each
        point.
        @func: The target function, i.e., the regression function.
        @size: The number of points generated from the function.
        @perturb: The precision of Gaussion noise on generated points.
        @[low, high): The range of x.
    '''
    assert precision > 0, 'Pertubation must be positive.'
    x = random.uniform(low=low, high=high, size=size)
    y = []
    if func == 'sin':
        y = np.sin(x)
    else:
        pass
    noise = random.normal(loc=0, scale=np.sqrt(1/precision), size=size)
    y = y + noise
    data = pd.DataFrame(data=zip(x, y), columns=['x', 'y'])
    return data

def designMatrixGenerate(dataset:DataFrame, basis:str='poly', order:int=10, positions=None, scaler=None):
    x = cp.array(list(dataset.iloc[:, 0]))
    if basis == 'poly':
        return _polyDesignMatrix(x, order)
    elif basis == 'gauss':
        if positions is None:
            print("No position parameters specified, use the random parameters instead...")
            # TODO: generate random position parameters.
        else:
            assert order == len(positions), "The order must match the number of position parameters!"
        if scaler is None:
            print("No scaler specified, use random one instead...")
            # TODO: generate random spatial scaler parameter.
        else:
            assert scaler>0, "Spatial scaler must be positive!"
        return _gaussDesignMatrix(x, order, positions, scaler)
    else:
        pass

def _polyDesignMatrix(x:cp.ndarray, order:int)->cp.ndarray:
    _len = len(x)
    designMatrix = cp.zeros([_len, order])
    _x = cp.ones(_len)
    for i in range(order):
        if i == 0:
            designMatrix[:, i] = _x
        else:
            designMatrix[:, i] = cp.power(x, i)
    return cp.asnumpy(designMatrix)

def _gaussDesignMatrix(x:cp.ndarray, order:int, positions:cp.ndarray, scaler:float):
    _len = len(x)
    designMatrix = cp.zeros([_len, order])
    for i in range(order):
        designMatrix[i, :] = gaussKernel(x, positions[i], scaler)
    return cp.asnumpy(designMatrix)

def LMS_epoch(w, dataset:pd.DataFrame, batch=1, lr=0.1, basis='poly', order=10, positions=None, scaler=None):
    '''
    In this function, we implement the Least-Mean-Square algorithm which is known as a
    sequential iteration progress. It requires the initial value @w of our parameter, and 
    we modify the parameter in-place.
    '''
    def LMS_iter(w, minibatch:pd.DataFrame):
        n = minibatch.shape[0]
        miniDesign = designMatrixGenerate(minibatch, basis, order, positions, scaler)
        t = np.array(minibatch.iloc[:, 1])
        sub = ((lr/batch)*t-np.matmul((lr/batch)*miniDesign, w)).T
        dw = np.matmul(sub, miniDesign)
        w += dw

    N = dataset.shape[0]
    for i in range(0, N, batch):
        seq = np.arange(i, i+batch)
        miniBatch = dataset.iloc[np.remainder(seq, N), :]
        LMS_iter(w, miniBatch)

def LMS(w, dataset:pd.DataFrame, batch=1, lr=1e-14, basis='poly', order=10, epsilon=1, positions=None, scaler=None):
    converge = False
    while converge == False:
        w_n = w.copy()
        LMS_epoch(w, dataset, batch, lr, basis, order, positions, scaler)
        if np.linalg.norm(w-w_n) < epsilon:
            # print(np.linalg.norm(w-w_n))
            converge = True