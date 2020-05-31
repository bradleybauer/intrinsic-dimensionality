from phEstimator import estimateIDPH0EstMST, estimateIDPH0ExactMST
from graphEstimator import estimateIDGraph

from loadData import loadDatasets


def testDistancePdfEstimator():
  pass
  # print('Testing Distance PDF Estimator')
  # datasets = loadDatasets()
  # compare approx and exact pdf estimate on each dataset
  # show results as table/plot


def testMstEstimator():
  # Load many different datasets
  # compare approx and exact mst estimate on each dataset
  # show results as table/plot
  pass


def testIdEstimators():
  datasets = loadDatasets()
  for name, dataset in datasets.items():
    exactMst = 'Timeout'
    if dataset.shape[0] < 5000:
      exactMst = estimateIDPH0ExactMST(dataset)
    estMst = estimateIDPH0EstMST(dataset)
    estGraph = estimateIDGraph(dataset)

    print('Dataset:', name, end='\t')
    print('PH0ExactMST:', exactMst, end='\t')
    print('PH0EstMST:', estMst, end='\t')
    print('GraphExactPDF:', 'not impl', end='\t')
    print('GraphEstPDF:', estGraph)
