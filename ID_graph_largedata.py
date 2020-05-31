import torchvision as tv
import math
import torch as th
from torchvision.models import resnet34
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cugraph
import cudf
import networkx as nx
import pickle
from scipy.optimize import curve_fit
from facenet_pytorch import InceptionResnetV1 as faceNet
from facenet_pytorch import MTCNN
from PIL import Image
import os
from multiprocessing import Process, Queue, Pool
import mlpack
import random
import concurrent.futures
import faiss

th.backends.cudnn.benchmark = True
th.backends.cudnn.enabled = True


# Create more difficult dataset from basic synthetic manifolds.
# Union hypersphere and hypercube.
# The idea is to be confident that the ID estimators work well on complex datasets.

# explore properties of knn graph. what is highest degree node?
# sort nodes according to degree
# then run sssp or fw on sorted knn graph.

# minimizing k makes sure that any path in the graph represents a smooth interpolation between points?

# given the shortest distances from u to all other nodes, we have that the shortest distance from v to any node r
# is less than (dvu + minDist(u,r)). However, minDist(u,r) may be large and so computing sssp on v may give a much
# better approximation to minDist(v,r). Maybe we can eliminate some computation by assuming that for the KNNs of u,
# their contribution to the pdf is not very important given u's contribution.
# willing to bet that for some unbounded quality measure, the quality of approximation grows logarithmically with
# sample size

# Since every subpath of a shortest path is a shortest path we could use the shortest path tree returned by sssp
# to get more samples

# if u,v are nn then the distances of u,v bound eachother. If u,v are close then calculating sssp for both u and v is
# basically redundant.


def loadData():
  print('Loading Dataset')

  # ############# MNIST #############
  # transform = tv.transforms.Compose([
  #   tv.transforms.ToTensor()
  # ])
  #
  # # Download training data
  # train = tv.datasets.MNIST(root='../datasets/MNIST', train=True, download=True, transform=transform)
  # # Download testing data
  # test = tv.datasets.MNIST(root='../datasets/MNIST', train=False, download=True, transform=transform)
  #
  # # Take only a single digit
  # digit = 2
  # idx = train.targets == digit
  # train.targets = train.targets[idx]
  # train.data = train.data[idx]
  # idx = test.targets == digit
  # test.targets = test.targets[idx]
  # test.data = test.data[idx]
  # dataset=test
  #
  # # dataset = th.utils.data.ConcatDataset([train, test])
  # dataset = th.utils.data.DataLoader(dataset, shuffle=True)
  # dataset = np.array(list(map(lambda x: x[0].numpy().ravel(), dataset)))  # discard the label
  # vis = np.reshape(dataset[0],(28,28))
  #
  # plt.matshow(vis)
  # plt.show()

  # print('Embedding Images')
  ############# LFW #############
  # # def processLFWSubset(filePathBatches, queue):
  # #   mtcnn = MTCNN()
  # #   resnet = faceNet(pretrained='vggface2', device=th.device('cuda')).eval()
  # #   embeddings = []
  # #   for filePathBatch in filePathBatches:
  # #     batch = [Image.open(f) for f in filePathBatch]
  # #     cropWhitenedBatch = th.cat([t.unsqueeze(0) for t in mtcnn(batch)]).to('cuda')
  # #     # TODO how is vggface2 different from pretrained='casia-webface' ? ? ?
  # #     embeddings.append(resnet(cropWhitenedBatch).cpu().numpy())
  # #   queue.put(embeddings)
  # # with th.no_grad():
  # #   filePaths = []
  # #   for root, dirs, files in os.walk('../datasets/lfw/lfw'):
  # #     filePaths.extend(root + '/' + f for f in files if f.endswith('.jpg'))
  # #   batchSize = 64
  # #   filePathBatches = [filePaths[i:i + batchSize] for i in range(0, len(filePaths), batchSize)]
  # #   numBatchesInSubset = 20
  # #   subsets = [filePathBatches[i:i + numBatchesInSubset] for i in range(0, len(filePathBatches), numBatchesInSubset)]
  # #   embeddings = []
  # #   for subset in tqdm(subsets):
  # #     q = Queue()
  # #     p = Process(target=processSubset, args=(subset, q))
  # #     p.start()
  # #     embeddings.extend(q.get())
  # #     p.join()
  # #   dataset = np.concatenate(embeddings)
  # #   np.save('../datasets/lfw/lfwEmbeddings.npy', dataset)
  # dataset = np.load('../datasets/lfw/lfwEmbeddings.npy')
  # np.random.shuffle(dataset)

  ############# ImageNet #############
  # # with th.no_grad():
  # #   normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
  # #                                       std=[0.229, 0.224, 0.225])
  # #   dataset = th.utils.data.DataLoader(
  # #     tv.datasets.ImageFolder('/home/xdaimon/ImageNet-Datasets-Downloader/imagenet',
  # #                             tv.transforms.Compose([
  # #                               tv.transforms.RandomSizedCrop(224),
  # #                               tv.transforms.ToTensor(),
  # #                               normalize,
  # #                             ])),
  # #     batch_size=64, shuffle=True, pin_memory=True)
  # #   print('Number of ImageNet images (contains dupes) ~ ', len(dataset)*64)
  # #   resnet = resnet34(pretrained=True, progress=True)
  # #   resnetExtractor = th.nn.Sequential(*list(resnet.children())[:-1]).eval().to('cuda')
  # #   embeddings = []
  # #   for xs, ys in tqdm(dataset):
  # #     embedding = resnetExtractor(xs.to('cuda')).cpu().numpy()
  # #     embedding = embedding.squeeze(2).squeeze(2)
  # #     embeddings.append(embedding)
  # # dataset = np.concatenate(embeddings)
  # # np.save('../datasets/imageNetEmbeddings.npy', dataset)
  # dataset = np.load('../datasets/imageNetEmbeddings.npy')
  # np.random.shuffle(dataset)

  # ####### Synthetics #######
  sets = ['Sinusoid', 'S', 'Gauss', 'Moebius', 'M12']
  intrinsicDims = [[1], [3, 5, 7, 9], [3, 4, 5, 6], [2], [12]]
  ambientDims = [[3], [4, 6, 8, 10], [3, 4, 5, 6], [3], [72]]
  subsetSizes = [[400, 500, 600], [600, 800, 1000, 1200], [100, 200, 400, 800], [20, 40, 80, 120],
                 [200, 400, 800, 1600]]
  whichSet = 4
  setName = sets[whichSet]
  print("Loading synthetic dataset.", )
  # Each file contains a dataset partitioned into 90 subsets
  numSubsets = 90
  # for intrinsicDim, ambientDim in list(zip(intrinsicDims[whichSet], ambientDims[whichSet])):
  # for subsetSize in subsetSizes[whichSet][::-1]:
  ambientDim = ambientDims[whichSet][-1]
  intrinsicDim = intrinsicDims[whichSet][-1]
  subsetSize = subsetSizes[whichSet][-1]
  print('Name:', setName)
  print('AmbientDim:', ambientDim)
  print('IntrinsicDim:', intrinsicDim)
  print('SubsetSize:', subsetSize)
  print('SubsetSize*NumSubsets:', subsetSize * numSubsets)

  # randomSubset = random.randint(0, numSubsets - 1)
  randomSubset = 10
  if len(ambientDims[whichSet]) > 1:
    filename = sets[whichSet] + str(intrinsicDim) + '_' + str(subsetSize) + '.BIN'
  else:
    filename = sets[whichSet] + '_' + str(subsetSize) + '.BIN'
  with open('../datasets/synthetic/data/' + filename, 'rb') as f:
    X = np.fromfile(f, dtype=np.float32)

    # TODO dont know how the data is stored in the file
    # is it this
    # X = np.reshape(X, (ambientDim, numSubsets * subsetSize))
    # or
    X = np.reshape(X, (numSubsets * subsetSize, ambientDim)).T
    # i think it is column major in memory so the second one should be right

    X = X[:, subsetSize * randomSubset:subsetSize * (randomSubset + 1)]
    dataset = np.ascontiguousarray(X.T)

  print('Dataset length:',len(dataset))
  return dataset


def getKnnRelation(K, X, gpu=True):
  """
  Returns the is kth nearest neighbor relation ... bradly todo comments
  :param K:
  :param data:
  :return:
  """
  # The first curve i found, invert that then take the derivative, is a bell curve.
  ambientDimension = X.shape[1]

  print('Building KNN Graph On GPU')
  if gpu:
    resources = faiss.StandardGpuResources()
  index = faiss.IndexFlatL2(ambientDimension)
  if gpu:
    index = faiss.index_cpu_to_gpu(resources, 0, index)
  index.add(X)
  distances, indices = index.search(X, K + 1)
  distances = np.sqrt(distances)  # faiss returns squared L2 norm
  print('Done!')

  # Store graph in COO sparse format
  graph_us = []
  graph_vs = []
  graph_ds = []
  for i, (nbrs, dists) in enumerate(zip(indices, distances)):
    graph_us.extend([i] * K)
    graph_vs.extend(nbrs[1:K + 1].tolist())
    graph_ds.extend(dists[1:K + 1].tolist())

  return graph_us, graph_vs, graph_ds

  # import ngtpy
  # if not os.path.exists('../datasets/MNIST/KnnIndex'):
  #   print('creating index')
  #   ngtpy.create(path='../datasets/MNIST/KnnIndex', dimension=784, edge_size_for_creation=k, edge_size_for_search=k)
  #   index = ngtpy.Index('../datasets/MNIST/KnnIndex') # open the index
  #   for x in tqdm(xb):
  #     index.insert(x.tolist())
  #   print('Building Index')
  #   index.build_index() # build index
  #   print('Saving Index')
  #   index.save() # save the index
  #   print('Done!')
  # else:
  #   index = ngtpy.Index('../datasets/MNIST/KnnIndex') # open the index
  # print('Creating knn graph')
  # graph = {}
  # for i,x in tqdm(enumerate(xb)):
  #   result = index.search(x.tolist(), k+1)
  #   neighbors = []
  #   for rank,(id,pdfMaxDist) in enumerate(result):
  #     if id == i:
  #       continue
  #     neighbors.append((id,pdfMaxDist))
  #   graph[i] = neighbors[:k]
  # print('Done!')
  # return graph


def FractionOfDistancesSeenForNumRunsOfSSSP(i, N):
  return (i * N - i * (i + 1) / 2) / (N * (N - 1) / 2)


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
  mstWeights = []
  dEstimates = []
  # choosing higher number of points to start at helps
  inputs = zip([X] * mstSamples, np.arange(10 // mstStep, mstSamples + 1) * mstStep)
  # pows = ((100 + math.e**np.arange(3,12.2,.05)).astype(int)).tolist()
  # inputs = zip([X] * len(pows), pows)
  with concurrent.futures.ProcessPoolExecutor(1) as executor:
    futures = {executor.submit(KnnMST, task) for task in inputs}

    for i, fut in tqdm(enumerate(concurrent.futures.as_completed(futures))):
      x, y = fut.result()
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


def estimateIDPH0(X):
  mstStep = 50
  mstSamples = len(X) // mstStep
  mstWeights = []
  dEstimates = []
  inputs = zip([X] * mstSamples, np.arange(1000 // mstStep, mstSamples + 1) * mstStep)
  with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = {executor.submit(EuclideanMST, task) for task in inputs}

    for i, fut in enumerate(concurrent.futures.as_completed(futures)):
      x, y = fut.result()
      mstWeights.append((x, y))
      if i < 2:
        continue

      ar = np.array(sorted(mstWeights))
      xs = ar[:, 0]
      ys = ar[:, 1]

      print("Fitting")
      coeffs = np.polyfit(np.log(xs), np.log(ys), 1)
      d = 1 / (1 - coeffs[0])
      print('Estimated d:', d)
      dEstimates.append(d)
      plt.scatter(xs, ys)
      plt.show()
      plt.plot(dEstimates)
      plt.show()


def func(x, a, b, c):
  return a * np.log(np.sin(x / 1 * np.pi / 2.))


def func2(x, a):
  return -a / 2. * (x - 1) ** 2


def func3(x, a, b, c):
  return np.exp(c) * np.sin(x / b * np.pi / 2.) ** a


def getIDforPDF(pdf, pdfMaxDist, numBins, K, ambientDimension):
  dx = pdfMaxDist / numBins

  pdfX = np.arange(numBins) * dx + dx / 2
  pdfMean = (pdf * pdfX).sum()
  pdfStd = np.sqrt((pdf * (pdfX - pdfMean) ** 2).sum())
  print('pdfMean:', pdfMean)
  print('pdfStd:', pdfStd)

  pdfMax = np.argmax(pdf) * dx
  print('pdfArgMax:', pdfMax)

  left_distr_x = pdfX[(pdfX > pdfMax - pdfStd) & (pdfX < pdfMax + pdfStd / 2.0)]
  left_distr_y = np.log(pdf[(pdfX > pdfMax - pdfStd) & (pdfX < pdfMax + pdfStd / 2.0)])
  coeff = np.polyfit(left_distr_x, left_distr_y, 2, full=False)
  a0 = coeff[0]
  b0 = coeff[1]

  fitMax = -b0 / a0 / 2.0
  fitStd = np.sqrt(abs(-1 / a0 / 2.))

  left_distr_x = pdfX[(pdfX > fitMax - fitStd) & (pdfX < fitMax + fitStd / 2.)]
  left_distr_y = np.log(pdf[(pdfX > fitMax - fitStd) & (pdfX < fitMax + fitStd / 2.)])
  coeff = np.polyfit(left_distr_x, left_distr_y, 2, full=False)
  a = coeff[0]
  b = coeff[1]
  c = coeff[2]

  fitMax = abs(-b / a / 2.)
  fitStd = np.sqrt(abs(-1 / a / 2.))
  fitMin = max(fitMax - 2 * np.sqrt(abs(-1 / a / 2.)) - dx / 2, 0.)

  rM = fitMax + dx / 4

  # 3 Gaussian Fitting to determine ratio R
  left_distr_x = pdfX[(pdfX > fitMin) & (pdfX <= rM) & (pdf > 0.000001)] / fitMax
  left_distr_y = np.log(pdf[(pdfX > fitMin) & (pdfX <= rM) & (pdf > 0.000001)]) - (4 * a * c - b ** 2) / 4. / a

  fit = curve_fit(func2, left_distr_x, left_distr_y)
  ratio = np.sqrt(fit[0][0])
  y1 = func2(left_distr_x, fit[0][0])
  # 3

  # 4 Geodesics D-Hypersphere Distribution Fitting to determine Dfit
  fit = curve_fit(func, left_distr_x, left_distr_y)
  Dfit = fit[0][0] + 1

  y2 = func(left_distr_x, fit[0][0], fit[0][1], fit[0][2])
  # 4

  # 5 Determination of Dmin
  res = np.empty(ambientDimension)
  for D in range(1, ambientDimension + 1):
    y = func(left_distr_x, D - 1, 1, 0)
    for i in range(0, len(y)):
      res[D - 1] = np.linalg.norm(y - left_distr_y) / np.sqrt(len(y))

  Dmin = np.argmax(-res) + 1

  y = func(left_distr_x, Dmin - 1, fit[0][1], 0)
  # 5

  # 6 Printing results
  print('FITTING PARAMETERS:')
  print('\t fitMax:', fitMax)
  print('\t fitStd:', fitStd)
  print('\t fitMin:', fitMin)
  print('FITTING RESULTS:')
  print('\t R:', ratio)
  print('\t Dfit:', Dfit)
  print('\t Dmin:', Dmin)

  plt.figure(2)
  plt.plot(left_distr_x, left_distr_y, 'o-', markersize=2, label='Representation (K=' + str(K) + ')')
  plt.plot(left_distr_x, y1, label='Gaussian (m={})'.format(int(Dmin)))
  plt.plot(left_distr_x, y2, label='Hypersphere (m={})'.format(int(Dmin)))
  plt.xlabel(r'$log (\frac{r}{r_{max}})$')
  plt.ylabel(r'$log (\frac{p(r)}{p(r_{max})})$')
  plt.legend()
  plt.grid(True)
  plt.show()

  plt.figure(3)
  plt.plot(res, 'o-', markersize=2, label='m (K=' + str(K) + ')')
  plt.xlabel('Dimension')
  plt.ylabel('Root Mean Squared Error')
  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)
  plt.grid(True)
  plt.show()


def getPDF(knnRelation, numBins, numSamples, numberOfNodes):
  us, vs, ds = map(cudf.Series, knnRelation)
  us, vs, ds = cugraph.structure.symmetrize(us, vs, ds)

  G = cugraph.Graph()
  G.add_edge_list(us, vs, ds)

  pdf = np.zeros(numBins)
  for i in tqdm(range(min(numSamples, numberOfNodes))):
    ssspResult: cudf.DataFrame = cugraph.sssp(G, i)
    distances: cudf.Series = ssspResult['distance']
    vertexIds: cudf.Series = ssspResult['vertex']
    # plt.scatter(range(len(distances)), sorted(distances))  # looks somewhat like an inverse sigmoid
    distances = distances[vertexIds > i]
    if i == 0:
      pdfMaxDist = 1.2 * distances.max()
    hist, _ = np.histogram(distances.tolist(), bins=numBins, range=(0, pdfMaxDist))
    pdf += hist
  return pdfMaxDist, pdf / pdf.sum()


def estimateIDGraph(X):
  K = 9
  ambientDimension = X.shape[1]
  numberOfNodes = X.shape[0]
  numBins = 100
  numSamples = 1000

  knnRelation = getKnnRelation(K, X)
  pdfMaxDist, pdf = getPDF(knnRelation, numBins, numSamples, numberOfNodes)

  # Show the estimated shortest distance pdf
  plt.hist(np.arange(numBins) / (numBins - 1) * pdfMaxDist, weights=pdf, bins=numBins)
  plt.show()

  getIDforPDF(pdf, pdfMaxDist, numBins, K, ambientDimension)


def main():
  X = loadData()
  print('Estimating Dimension')
  # estimateIDPH0(X)
  # estimateIDPH0EstMST(X)
  estimateIDGraph(X)


if __name__ == '__main__':
  main()
