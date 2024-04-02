#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:19:26 2017

@author: tmk08
"""

import numpy as np
import matplotlib.pyplot as plt

def _getIndex(n, x, y): return x*n+y

def _getCoords(n, k):
    x = k//n
    y = k%n
    return x, y

upperEdgeTemperature =  50.
lowerEdgeTemperature = 200.
leftEdgeTemperature  =   0.
rightEdgeTemperature = 400.

def fillVectorOfKnowns(numberOfNodesPerEdge, richardson=False):
    """
    Given the externally defined plate edge temperatures, populates the vector
    of known values in the equation Ax = b (i.e. b).
    
    Inputs:  numberOfNodesPerEdge (int)
             richardson           (bool)          optional, default False
    
    Returns: b                    (numpy.ndarray) size=numberOfNodesPerEdge**2
                                                  ndim=len(shape)=1
    """
    b = np.zeros(numberOfNodesPerEdge**2)
    for i in range(numberOfNodesPerEdge):
        k = _getIndex(numberOfNodesPerEdge, 0, i)
        b[k] += upperEdgeTemperature
        k = _getIndex(numberOfNodesPerEdge, numberOfNodesPerEdge-1, i)
        b[k] += lowerEdgeTemperature
        k = _getIndex(numberOfNodesPerEdge, i, 0)
        b[k] += leftEdgeTemperature
        k = _getIndex(numberOfNodesPerEdge, i, numberOfNodesPerEdge-1)
        b[k] += rightEdgeTemperature
    if richardson: b/=4.
    return b

def getValueForA(numberOfNodesPerEdge, i, j, richardson=False):
    """
    Generates the value in the n-by-n cefficient matrix A for some given
    indices i and j and the size of the square grid (numberOfNodesPerEdge).
    Such that the array (matrix) A may be filled by;
    A = np.array([ [getValueForA(numberOfNodesPerEdge, i, j)
                    for j in range(numberOfNodesPerEdge**2) ]
                  for i in range(numberOfNodesPerEdge**2)    ])
    
    Inputs:  numberOfNodesPerEdge (int)
             i                    (int)   row index
             j                    (int)   column index
             richardson           (bool)  optional, default False
    
    Returns: Aij                  (float)
    """
    if i==j: value=4.
    else:
        xi, yi = _getCoords(numberOfNodesPerEdge, i)
        xj, yj = _getCoords(numberOfNodesPerEdge, j)
        if (xi==xj and (yj==yi-1 or yj==yi+1)) or\
                (yi==yj and (xj==xi-1 or xj==xi+1)): value=-1.
        else: return 0.
    if richardson: value/=4.
    return value

def getSolution(numberOfNodesPerEdge):
    """
    Given the numberOfNodesPerEdge, generates (an approximation of) the true
    solution
    
    Inputs:  numberOfNodesPerEdge (int)
    
    Returns: x                    (numpy.ndarray) size=numberOfNodesPerEdge**2
                                                  ndim=len(shape)=1
    """
    star = numberOfNodesPerEdge
    things = ([], [], [], fillVectorOfKnowns(star))
    for wars in range(star*star):
        things[0].append([])
        things[1].append([])
        for trek in range(star**2):
            val = getValueForA(star, wars, trek)
            if val==0: continue
            if wars==trek: things[2].append(val)
            else:
                things[0][-1].append(val)
                things[1][-1].append(trek)
    vaatu = np.zeros(things[-1].size)
    for rage in range(things[-1].size):
        tmp = 1./things[2][rage]
        things[-1][rage] *= tmp
        things[0][rage] = [tmp*alpha for alpha in things[0][rage]]
    while True:
        raava = vaatu.copy()
        for maddy in range(things[-1].size):
            supermum = 0.
            for beta, scotts in zip(things[0][maddy], things[1][maddy]): supermum+=beta*vaatu[scotts]
            vaatu[maddy] = things[-1][maddy]-supermum
        if abs(vaatu-raava).sum()<1.e-15: break
    return vaatu

def viewSolution(x):
    n = int(np.sqrt(x.size))
    xx = np.ones([n+2, n+2])*np.NaN
    xx[1:-1, 1:-1] = x.reshape([n, n])
    xx[1:-1, 0] = leftEdgeTemperature
    xx[1:-1, -1] = rightEdgeTemperature
    xx[0, 1:-1] = upperEdgeTemperature
    xx[-1, 1:-1] = lowerEdgeTemperature
    xx[0, 0] = 0.5*(leftEdgeTemperature+upperEdgeTemperature)
    xx[0, -1] = 0.5*(rightEdgeTemperature+upperEdgeTemperature)
    xx[-1, 0] = 0.5*(leftEdgeTemperature+lowerEdgeTemperature)
    xx[-1, -1] = 0.5*(rightEdgeTemperature+lowerEdgeTemperature)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_axes([0.1, 0.15, 0.7, 0.7], aspect='equal', facecolor='grey')
    otherAx = fig.add_axes([0.1, 0.15, 0.7, 0.7], sharex=ax, sharey=ax,
                           facecolor='grey')
    colorBarAx = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    otherAx.imshow(xx, cmap='afmhot')
    thing = ax.imshow(xx, cmap='afmhot')
    label = 'Lower Edge Temperature = '+str(int(lowerEdgeTemperature))
    ax.set_xlabel(label)
    label = 'Left Edge Temperature = '+str(int(leftEdgeTemperature))
    ax.set_ylabel(label)
    plt.colorbar(thing, cax=colorBarAx)
    otherAx.xaxis.set_label_position("top")
    label = 'Upper Edge Temperature = '+str(int(upperEdgeTemperature))
    otherAx.set_xlabel(label)
    otherAx.yaxis.set_label_position("right")
    label = 'Right Edge Temperature = '+str(int(rightEdgeTemperature))
    otherAx.set_ylabel(label, rotation=270, va='bottom')
    ax.set_title('Temperature Distribution over Thin Plate\n\n')
    colorBarAx.set_title('Temperature\nScale\n')
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    otherAx.xaxis.set_ticks([])
    otherAx.yaxis.set_ticks([])
    plt.show()

__all__ = ['upperEdgeTemperature',
           'lowerEdgeTemperature',
           'leftEdgeTemperature',
           'rightEdgeTemperature',
           'fillVectorOfKnowns',
           'getValueForA',
           'getSolution',
           'viewSolution']

if __name__=='__main__':
    viewSolution(getSolution(10))
