from knn import getKnnRelation
import numpy as np


def idIDEA(X, K):
    distances, indices = getKnnRelation(K, X, inCOOFormat=False)
    N = X.shape[0]
    # distances.shape = (N,K+1)
    distances /= distances[:, K, None]
    m = distances.sum() / (N * K)
    return float(m) / (1-m)
