from loadData import loadDatasets
from experiments import *


def main():
    datasets, labels = loadDatasets()

    # testDeepMdsPreservesID()

    # testAdvAttackImageNet()

    # testDistancePdfEstimator(datasets)

    # testMstEstimator(datasets)

    testIdEstimators(datasets)


if __name__ == '__main__':
    main()
