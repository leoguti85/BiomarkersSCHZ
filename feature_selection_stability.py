# Feature selection stability index
# This script is based on "A stability index for feature selection", Kuncheva, 2007

from __future__ import division # To get floating point division
import numpy as np

def pairwise_consistency_index(Si,Sj,n):
    k = Si.shape[0] # Size of the reduced feature vector
    r = np.intersect1d(Si,Sj).shape[0]
    index = (r*n - pow(k,2))/(k*(n-k))
    return index

def Kuncheva_index(sequences,n):
    K = sequences.shape[0] # Number of sequences
    index = 0
    for i in range(0,K-2):
        for j in range(i+1,K-1):
            index = index + pairwise_consistency_index(sequences[i,:],sequences[j,:],n)
    K_index = (2*index)/(K*(K-1))
    return K_index
