import torchvision as tv
import torch as th
import numpy as np
import networkx as nx
import faiss
import cudf
import mlpack
import random


def getKnnRelation(k, X):
  ambientDimension = X.shape[1]

  # print('Building KNN Graph for k=' + str(k))
  resources = faiss.StandardGpuResources()
  index = faiss.IndexFlatL2(ambientDimension)
  index = faiss.index_cpu_to_gpu(resources, 0, index)
  index.add(X)
  distances, indices = index.search(X, k + 1)
  distances = np.sqrt(distances)  # faiss returns squared L2 norm
  # print('Done!')

  # Store graph in COO sparse format
  graph_us = []
  graph_vs = []
  graph_ds = []
  for i, (nbrs, dists) in enumerate(zip(indices, distances)):
    graph_us.extend([i] * k)
    graph_vs.extend(nbrs[1:k + 1].tolist())
    graph_ds.extend(dists[1:k + 1].tolist())

  return graph_us, graph_vs, graph_ds


# Thick metal plate, with holes punched through it.


# Changes the MST weight should NOT increase across iterations.
# If it does then this may be due to non commutativity of floating point addition.

# MNIST
# exact = 14957.282871772331
# approx = 14957.302264034748

# LFW
# exact = 9100.169333803324
# approx = 9100.26 using k=25 and 9100.21 for k=39
# approx remains 9100.211251 ... for k from 39 to 65
# at k = 66 we get ~9100.16966 and stay here through 99

# ImageNet
# approx =
# 3118154.5337726837
# 3118152.810012079
# 3118151.925573564
# 3118148.537595964
# 3118147.0783071285
# 3118147.078305221
# 3118147.078303314
# 3118147.078300453
# 3118147.078297592
# 3118147.0782956844
# ...
# 3118147.0783021217 for  k = 92

def estimateMstWeight(X):
  # print('Number of points:', len(X))
  # for k in range(len(X)-1, len(X)-1 + 1):
  # for k in range(10,20):
  # np.random.shuffle(X)

  k = 20
  # print('Getting KNN graph')
  us, vs, ds = getKnnRelation(k, X)  # u ~ v with strength 1/d? lol

  # TODO can probably find a faster mst implementation

  # print('Solving for MST in KNN graph. ',end='')
  knnGraph = nx.Graph()
  knnGraph.add_weighted_edges_from(zip(us, vs, ds))  # G is undirected, so adding both directions is unneccessary
  mstEdges = nx.minimum_spanning_edges(knnGraph, algorithm='boruvka')
  # mstEdges = nx.minimum_spanning_edges(knnGraph, algorithm='kruskal')

  # print('Unpacking edge weights. ',end='')
  _, _, edgeWeights = zip(*mstEdges)  # unzip the list of tuples
  # print('Weight sum: ', end='')
  mstWeight = np.array([w['weight'] for w in edgeWeights]).sum()
  # print(mstWeight)
  return mstWeight


# mstEdges = mlpack.emst(X, copy_all_inputs=True, leaf_size=1, naive=False)['output']
# mstWeight = mstEdges[:, 2].sum()
# print(mstWeight)


def loadData():
  ############ MNIST #############
  # transform = tv.transforms.Compose([
  #   tv.transforms.ToTensor()
  # ])
  #
  # # Download training data
  # train = tv.datasets.MNIST(root='../datasets/MNIST', train=True, download=True, transform=transform)
  # # Download testing data
  # test = tv.datasets.MNIST(root='../datasets/MNIST', train=False, download=True, transform=transform)
  #
  # # Take only the handwritten twos
  # idx = train.targets == 1
  # train.targets = train.targets[idx]
  # train.data = train.data[idx]
  # idx = test.targets == 1
  # test.targets = test.targets[idx]
  # test.data = test.data[idx]
  #
  # dataset = th.utils.data.ConcatDataset([train, test])
  # dataset = th.utils.data.DataLoader(dataset, shuffle=True)
  # dataset = np.array(list(map(lambda x: x[0].numpy().ravel(), dataset)))  # discard the label

  # #### LFW ###
  # dataset = np.load('../datasets/lfw/lfwEmbeddings.npy')
  # np.random.shuffle(dataset)

  #### ImageNet ####
  # dataset = np.load('../datasets/imageNetEmbeddings.npy')
  # np.random.shuffle(dataset)

  #### Synthetics ####
  sets = ['Sinusoid', 'S', 'Gauss', 'Moebius', 'M12']
  intrinsicDims = [[1], [3, 5, 7, 9], [3, 4, 5, 6], [2], [12]]
  ambientDims = [[3], [4, 6, 8, 10], [3, 4, 5, 6], [3], [72]]
  subsetSizes = [[400, 500, 600], [600, 800, 1000, 1200], [100, 200, 400, 800], [20, 40, 80, 120],
                 [200, 400, 800, 1600]]

  whichSet = 0
  setName = sets[whichSet]

  # Each file contains a dataset partitioned into 90 subsets
  numSubsets = 90
  for intrinsicDim, ambientDim in zip(intrinsicDims[whichSet], ambientDims[whichSet]):
    for subsetSize in subsetSizes[whichSet]:
      print('Name:', setName)
      print('AmbientDim:', ambientDim)
      print('IntrinsicDim:', intrinsicDim)
      print('SubsetSize:', subsetSize)
      print('SubsetSize*NumSubsets:', subsetSize * numSubsets)

      # randomSubset = random.randint(0, numSubsets - 1)
      randomSubset = 3
      if len(ambientDims[whichSet]) > 1:
        filename = sets[whichSet] + str(intrinsicDim) + '_' + str(subsetSize) + '.BIN'
      else:
        filename = sets[whichSet] + '_' + str(subsetSize) + '.BIN'
      with open('../datasets/synthetic/data/' + filename, 'rb') as f:
        X = np.fromfile(f, dtype=np.float32)
        print()
        print(X.shape)
        X = np.reshape(X, (ambientDim, numSubsets * subsetSize))
        print(X.shape)
        X = X[:, subsetSize * randomSubset:subsetSize * (randomSubset + 1)]
        print(X.shape)
        print()
        # For now just return one thing
        return np.ascontiguousarray(X.T)

  return dataset


def main():
  X = loadData()
  estimateMstWeight(X)


if __name__ == '__main__':
  main()
