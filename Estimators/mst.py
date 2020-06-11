import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import mlpack

from knn import getKnnRelation


def knnMstWeight(X, K=20):
    us, vs, ds = getKnnRelation(K, X)

    knnGraph = nx.Graph()
    knnGraph.add_weighted_edges_from(zip(us, vs, ds))  # G is undirected, so adding both directions is unneccessary
    mstEdges = nx.minimum_spanning_edges(knnGraph, algorithm='kruskal')

    _, _, edgeWeights = zip(*mstEdges)  # unzip the list of tuples
    return np.array([w['weight'] for w in edgeWeights]).sum()


def euclideanMstWeight(X):
    mstEdges = mlpack.emst(X, copy_all_inputs=True, leaf_size=1, naive=False)['output']
    mstWeight = mstEdges[:, 2].sum()
    return mstWeight


def idMst(X):
    return _estimateID(X, knnMstWeight)


def idMstExact(X):
    return _estimateID(X, euclideanMstWeight)


def _estimateID(X, getMstWeight):
    # increase either of these to get better estimation
    numSamples = 100
    numIters = 1

    logN = np.log(X.shape[0])
    if logN <= 4:
        print("Dataset too small")
        return 'Invalid'

    X = X[:]
    ns = np.exp(np.arange(4, logN, (logN - 4) / numSamples)).astype(int).tolist()

    estimates = []
    for j in range(numIters):
        mstWeights = []
        idEstimates = []
        for i, n in enumerate(ns):
            np.random.shuffle(X)
            mstWeights.append(getMstWeight(X[:n]))
            if i == 0:
                continue

            xs = np.array(ns[:len(mstWeights)])
            ys = np.array(mstWeights)
            slope = np.polyfit(np.log(xs), np.log(ys), deg=1)[0]
            d = 1 / (1 - slope)
            idEstimates.append(d)
            if i > 20 and np.std(idEstimates[-20:]) < .5:
                break
        estimates += [np.average(idEstimates[-20:])]

    # plt.scatter(xs, ys)
    # plt.show()
    # plt.plot(idEstimates)
    # plt.show()
    # print('', end='', flush=True)

    return str(np.mean(estimates))  # + '  std:' + str(np.std(estimates))
