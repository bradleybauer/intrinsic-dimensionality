from phEstimator import estimateIDPH0EstMST, estimateIDPH0ExactMST, knnMstWeight, euclideanMstWeight
from graphEstimator import estimateIDGraph, getKnnRelation, getPDF
from fisherEstimator import estimateIDFisher
import numpy as np


def testDistancePdfEstimator(datasets):
  print('Testing Distance PDF Estimator')
  for name, data in datasets.items():
    print('Dataset:', name)

    numberOfNodes = data.shape[0]
    numBins = max(min(numberOfNodes // 25, 1000), 100)
    numSamplesSmall = 25
    numSamplesBig = min(300, len(data))
    K = 12

    knnRelation = getKnnRelation(K, data)
    estPDF = getPDF(knnRelation, numBins, numSamplesSmall, numberOfNodes)[1]
    fullPDF = getPDF(knnRelation, numBins, numSamplesBig, numberOfNodes)[1]
    error = np.linalg.norm(estPDF - fullPDF)
    print('\tError:', error)


def testMstEstimator(datasets):
  print('Testing MST Estimators')
  print('Approx with K=20 vs Exact')
  for name, data in datasets.items():
    print('Dataset:', name, ' Size:', len(data))

    size = min(len(data), 3000)
    print('\tSubset size:', size)
    X = data[:size]
    np.random.shuffle(X)

    approxWeight = knnMstWeight(X, K=20)
    print('\t\tApprox:', approxWeight)

    exactWeight = euclideanMstWeight(X)
    print('\t\tExact :', exactWeight)

  print('Approx with K=20 vs Approx with K=100')
  for name, data in datasets.items():
    print('Dataset:', name, ' Size:', len(data))

    size = min(len(data), 100000)
    print('\tSubset size:', size)
    X = data[:size]
    np.random.shuffle(X)

    k20 = knnMstWeight(X, K=20)
    print('\t\tApproxK20 :', k20)
    k100 = knnMstWeight(X, K=100)
    print('\t\tApproxK100:', k100)


def testIdEstimators(datasets):
  for name, dataset in datasets.items():
    print('Testing estimators on', name)

    print('\tPH0ExactMST: ', end='')
    exactMst = estimateIDPH0ExactMST(dataset)
    print(exactMst)

    print('\tPH0EstMST: ', end='')
    estMst = estimateIDPH0EstMST(dataset)
    print(estMst)

    print('\tGraphEst: ', end='')
    estGraph = estimateIDGraph(dataset)
    print(estGraph)

    print('\tFisherEst: ', end='')
    estFisher = estimateIDFisher(dataset[:10000])
    print(estFisher)

    print()
    print()
