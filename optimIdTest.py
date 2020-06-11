from util import rprint
import networkx as nx
import numpy as np
import mlpack
import pickle
import torch as th
from loadData import loadImageNetEmbeddings, loadMNIST
from knn import getKnnRelation
from DeepMDS.model import DeepMDS

# TODO probably could use this differentiable ID est in Estimators.mst instead
# of the current numpy implementation. I think it is just as fast?

# TODO test on datasets other than imagenet

# TODO compare estimate of idGraph while optimizing the estimate of idMst

targetID = 9
K = 20

X, _ = loadImageNetEmbeddings()
X = th.cuda.FloatTensor(X)
X = X[:40000]

N = len(X)

with open('DeepMDS/layerSizes.pkl', 'rb') as f:
    layerSizes = pickle.load(f)
model = DeepMDS(layerSizes)
model.load_state_dict(th.load('DeepMDS/weights.pt'))
model = model.cuda()

optimizer = th.optim.Adam(model.parameters())
# optimizer = th.optim.SGD(model.parameters(), .1)

batchSize = 32
numBatches = len(X) // batchSize
remainder = len(X) % batchSize

while True:
    # Transform dataset
    transformedData = []
    for b in range(numBatches):
        batch = X[b * batchSize:(b + 1) * batchSize]
        transformedData.append(model(batch))
    if remainder:
        transformedData.append(model(X[-remainder:]))
    TX = th.cat(transformedData)

    mstWeights = []
    idEstimates = []
    numSamples = 70
    logN = np.log(TX.shape[0])
    ns = np.exp(np.arange(4, logN, (logN - 4) / numSamples)).astype(int).tolist()
    for i, n in enumerate(ns):
        TXshuffled = TX[th.randperm(N)]
        subset = TXshuffled[:n]

        # Calculate MST edge distances
        us, vs, ds = getKnnRelation(K, subset.detach().cpu().numpy())
        knnGraph = nx.Graph()
        knnGraph.add_weighted_edges_from(zip(us, vs, ds))  # G is undirected, so adding both directions is unneccessary
        mstEdges = nx.minimum_spanning_edges(knnGraph, algorithm='prim', data=False)
        us, vs = map(list, zip(*mstEdges))
        # TODO does us,vs have two directed edges to represent one undirected edge?
        distances = th.norm(subset[us] - subset[vs], dim=1)
        mstWeights.append(distances.sum())
        mstWeightVec = th.cat([v.view(1) for v in mstWeights])
        # print('mstWeights', mstWeights)
        # print('mstWeightVec', mstWeightVec)

        # Calculate id estimate
        logxs = th.log(th.cuda.FloatTensor(ns[:len(mstWeights)]))  # , requires_grad=False
        logys = th.log(mstWeightVec)
        # solve normal equation
        logxs = th.cat([logxs[:, None], th.ones_like(logxs[:, None])], dim=1)
        w = (th.inverse(logxs.T @ logxs) @ logxs.T) @ logys
        slope = w[0]
        d = 1 / (1 - slope)
        # print('logxs', logxs)
        # print('logys', logys)
        # print('w', w)
        # print('calc', logxs @ w)
        # print('d', d)
        idEstimates.append(d.view(1))
        # if i > 20 and th.std(th.cat(idEstimates[-20:])) < .5:
        #     break
        # print()
        # print()
        # print()
        # print()
        rprint(i)

    finalEst = th.mean(th.cat(idEstimates[-numSamples // 2:]))
    print('\nid estimate:', float(finalEst))
    err = (finalEst - targetID) ** 2
    optimizer.zero_grad()
    err.backward()
    optimizer.step()
    # for p in model.parameters():
    #     print(p.grad.data.norm())
