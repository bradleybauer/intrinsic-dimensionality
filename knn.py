import numpy as np
import faiss


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
