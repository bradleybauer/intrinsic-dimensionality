import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import numpy as np
import mlpack
import math

from knn import getKnnRelation


def estimateMstWeight(X):
  # print('Number of points:', len(X))
  # for k in range(len(X)-1, len(X)-1 + 1):
  # for k in range(10,20):
  # np.random.shuffle(X)

  K = 9
  # print('Getting KNN graph')
  us, vs, ds = getKnnRelation(K, X)  # u ~ v with strength 1/d? lol

  # TODO can probably find a faster mst implementation

  # print('Solving for MST in KNN graph. ',end='')
  knnGraph = nx.Graph()
  knnGraph.add_weighted_edges_from(zip(us, vs, ds))  # G is undirected, so adding both directions is unneccessary
  mstEdges = nx.minimum_spanning_edges(knnGraph, algorithm='kruskal')

  # print('Unpacking edge weights. ',end='')
  _, _, edgeWeights = zip(*mstEdges)  # unzip the list of tuples
  # print('Weight sum: ', end='')
  mstWeight = np.array([w['weight'] for w in edgeWeights]).sum()
  return mstWeight


def KnnMST(x):
  data, numPointsToUse = x
  np.random.shuffle(data)
  mstWeight = estimateMstWeight(data[:numPointsToUse])
  return numPointsToUse, mstWeight


def EuclideanMST(x):
  data, numPointsToUse = x
  np.random.shuffle(data)
  mstEdges = mlpack.emst(data[:numPointsToUse], copy_all_inputs=True, leaf_size=1, naive=False)['output']
  mstWeight = mstEdges[:, 2].sum()
  return numPointsToUse, mstWeight


def estimateIDPH0EstMST(X):
  mstStep = 10
  mstSamples = len(X) // mstStep
  useGPU = True
  inputs = ((100 + math.e ** np.arange(3, 12.2, .05)).astype(int)).tolist()
  inputs = zip([X] * len(inputs), inputs)
  return estimateID(X, mstStep, mstSamples, KnnMST, inputs, useGPU)


def estimateIDPH0ExactMST(X):
  mstStep = 10
  mstSamples = len(X) // mstStep
  useGPU = True
  inputs = zip([X] * mstSamples, np.arange(10 // mstStep, mstSamples + 1) * mstStep)
  return estimateID(X, mstStep, mstSamples, EuclideanMST, useGPU)


def estimateID(X, mstStep, mstSamples, getMST, inputs, useGPU):
  mstWeights = []
  dEstimates = []
  for i, input in tqdm(enumerate(inputs)):
    x, y = getMST(input)
    mstWeights.append((x, y))
    if i < 2:
      continue

    ar = np.array(sorted(mstWeights))
    xs = ar[:, 0]
    ys = ar[:, 1]
    print("Fitting. ", end='')
    coeffs = np.polyfit(np.log(xs), np.log(ys), 1)
    d = 1 / (1 - coeffs[0])
    dEstimates.append(d)
    print('Estimated d:', d)
    if i % 20 == 0:
      plt.scatter(xs, ys)
      plt.show()
      plt.plot(dEstimates)
      plt.show()
