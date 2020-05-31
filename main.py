from experiments import *

# Create more difficult dataset from basic synthetic manifolds.
# Union hypersphere and hypercube.
# The idea is to be confident that the ID estimators work well on complex datasets.

# explore properties of knn graph. what is the highest degree node?
# is the sum of sssp distances smaller for nodes that are in more dense areas?
# if we uniformly randomly choose nodes to compute sssp for, then we are more likely to choose nodes from highly dense
# areas.

# minimizing K makes sure that any path in the graph represents a smooth interpolation between points?

# given the shortest distances from u to all other nodes, we have that the shortest distance from v to any node r
# is less than (dvu + minDist(u,r)). However, minDist(u,r) may be large and so computing sssp on v may give a much
# better approximation to minDist(v,r). Maybe we can eliminate some computation by assuming that for the KNNs of u,
# their contribution to the pdf is not very important given u's contribution.
# willing to bet that for some unbounded quality measure, the quality of approximation grows logarithmically with
# sample size

# if u,v are nn then the distances of u,v bound eachother. If u,v are close then calculating sssp for both u and v is
# basically redundant.


def main():
  testDistancePdfEstimator()

  testMstEstimator()

  testIdEstimators()


if __name__ == '__main__':
  main()
