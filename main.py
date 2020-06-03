from loadData import loadDatasets
from experiments import *

def main():
  datasets = loadDatasets()

  # testDistancePdfEstimator(datasets)

  # testMstEstimator(datasets)

  testIdEstimators(datasets)


if __name__ == '__main__':
  main()

