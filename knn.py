import numpy as np
import faiss


def getKnnRelation(K, X):
    ambientDimension = X.shape[1]

    # print('Building KNN Graph On GPU')
    resources = faiss.StandardGpuResources()
    resources.setCudaMallocWarning(False)
    index = faiss.IndexFlatL2(ambientDimension)
    index = faiss.index_cpu_to_gpu(resources, 0, index)
    index.add(X)
    distances, indices = index.search(X, K + 1)
    distances = np.sqrt(distances)  # faiss returns squared L2 norm
    # print('Done!')

    # Store graph in COO sparse format
    graph_us = []
    graph_vs = []
    graph_ds = []
    for i, (nbrs, dists) in enumerate(zip(indices, distances)):
        graph_us.extend([i] * K)
        graph_vs.extend(nbrs[1:K + 1].tolist())
        graph_ds.extend(dists[1:K + 1].tolist())

    return graph_us, graph_vs, graph_ds
