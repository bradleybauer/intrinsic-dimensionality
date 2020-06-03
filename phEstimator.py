import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import mlpack

from knn import getKnnRelation


# TODO can probably find a faster mst implementation

def knnMstWeight(data, K=20):
  # print('Number of points:', len(X))
  # for k in range(len(X)-1, len(X)-1 + 1):
  # for k in range(10,20):
  # np.random.shuffle(X)

  us, vs, ds = getKnnRelation(K, data)  # u ~ v with strength 1/d? lol

  knnGraph = nx.Graph()
  knnGraph.add_weighted_edges_from(zip(us, vs, ds))  # G is undirected, so adding both directions is unneccessary
  mstEdges = nx.minimum_spanning_edges(knnGraph, algorithm='kruskal')

  _, _, edgeWeights = zip(*mstEdges)  # unzip the list of tuples
  return np.array([w['weight'] for w in edgeWeights]).sum()


def euclideanMstWeight(data):
  mstEdges = mlpack.emst(data, copy_all_inputs=True, leaf_size=1, naive=False)['output']
  mstWeight = mstEdges[:, 2].sum()
  return mstWeight


def estimateIDPH0EstMST(X):
  return estimateID(X, knnMstWeight)


def estimateIDPH0ExactMST(X):
  return estimateID(X, euclideanMstWeight)


def estimateID(X, getMstWeight):
  X = X[:]
  numSamples = 100
  logN = np.log(X.shape[0])
  if logN <= 4:
    print("Dataset too small")
    return 'Invalid'
  ns = np.exp(np.arange(4, logN, (logN - 4) / numSamples)).astype(int).tolist()
  weights = []
  idEstimates = []
  for i, n in enumerate(ns):
    np.random.shuffle(X)
    weights.append(getMstWeight(X[:n]))
    if i == 0:
      continue

    xs = np.array(ns[:len(weights)])
    ys = np.array(weights)
    slope = np.polyfit(np.log(xs), np.log(ys), deg=1)[0]
    d = 1 / (1 - slope)
    idEstimates.append(d)
    if i > 20 and np.std(idEstimates[-20:]) < .7:
      break

  plt.scatter(xs, ys)
  plt.show()
  plt.plot(idEstimates)
  plt.show()
  print('', end='', flush=True)
  return np.average(idEstimates[-20:])
