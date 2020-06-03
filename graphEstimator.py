from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cugraph
import cudf

from knn import getKnnRelation


def FractionOfDistancesSeenForNumRunsOfSSSP(i, N):
  return (i * N - i * (i + 1) / 2) / (N * (N - 1) / 2)


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
  # print('pdfMean:', pdfMean)
  # print('pdfStd:', pdfStd)

  pdfMax = np.argmax(pdf) * dx
  # print('pdfArgMax:', pdfMax)

  left_distr_x = pdfX[(pdfX > pdfMax - pdfStd) & (pdfX < pdfMax + pdfStd / 2.0)]
  left_distr_y = np.log(pdf[(pdfX > pdfMax - pdfStd) & (pdfX < pdfMax + pdfStd / 2.0)])
  coeff = np.polyfit(left_distr_x, left_distr_y, 2, full=False)
  a0 = coeff[0]
  b0 = coeff[1]

  fitMax = -b0 / a0 / 2.0
  fitStd = np.sqrt(abs(-1 / a0 / 2.))

  left_distr_x = pdfX[(pdfX > fitMax - fitStd) & (pdfX < fitMax + fitStd / 2.)]
  left_distr_y = np.log(pdf[(pdfX > fitMax - fitStd) & (pdfX < fitMax + fitStd / 2.)])
  try:
    coeff = np.polyfit(left_distr_x, left_distr_y, 2, full=False)
  except:
    return 'NA'
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
  try:
    fit = curve_fit(func2, left_distr_x, left_distr_y)
  except:
    return 'NA'
  # ratio = np.sqrt(fit[0][0])
  y1 = func2(left_distr_x, fit[0][0])
  # 3

  # 4 Geodesics D-Hypersphere Distribution Fitting to determine Dfit
  try:
    fit = curve_fit(func, left_distr_x, left_distr_y)
  except:
    return 'NA'
  # Dfit = fit[0][0] + 1

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
  # print('FITTING PARAMETERS:')
  # print('\t fitMax:', fitMax)
  # print('\t fitStd:', fitStd)
  # print('\t fitMin:', fitMin)
  # print('FITTING RESULTS:')
  # print('\t R:', ratio)
  # print('\t Dfit:', Dfit)
  # print('\t Dmin:', Dmin)

  plt.figure(2)
  plt.plot(left_distr_x, left_distr_y, 'o-', markersize=2, label='Representation (K=' + str(K) + ')')
  plt.plot(left_distr_x, y1, label='Gaussian (m={})'.format(int(Dmin)))
  plt.plot(left_distr_x, y2, label='Hypersphere (m={})'.format(int(Dmin)))
  plt.xlabel(r'$log (\frac{r}{r_{max}})$')
  plt.ylabel(r'$log (\frac{p(r)}{p(r_{max})})$')
  plt.legend()
  plt.grid(True)
  plt.show()

  # plt.figure(3)
  # plt.plot(res, 'o-', markersize=2, label='m (K=' + str(K) + ')')
  # plt.xlabel('Dimension')
  # plt.ylabel('Root Mean Squared Error')
  # plt.xticks(fontsize=15)
  # plt.yticks(fontsize=15)
  # plt.grid(True)
  # plt.show()

  return Dmin


def getPDF(knnRelation, numBins, numSamples, numberOfNodes):
  us, vs, ds = map(cudf.Series, knnRelation)
  us, vs, ds = cugraph.structure.symmetrize(us, vs, ds)
  df = cudf.DataFrame({'source': us, 'destination': vs, 'weight': ds})

  G = cugraph.Graph()
  G.from_cudf_edgelist(df, edge_attr='weight')

  pdf = np.zeros(numBins)
  # for i in tqdm(range(min(numSamples, numberOfNodes))):
  for i in range(min(numSamples, numberOfNodes)):
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
  K = 12
  ambientDimension = X.shape[1]
  numberOfNodes = X.shape[0]
  numBins = max(min(numberOfNodes // 25, 1000), 100)
  numSamples = 100

  knnRelation = getKnnRelation(K, X)
  pdfMaxDist, pdf = getPDF(knnRelation, numBins, numSamples, numberOfNodes)

  # Show the estimated shortest distance pdf
  plt.hist(np.arange(numBins) / (numBins - 1) * pdfMaxDist, weights=pdf, bins=numBins)
  plt.show()

  return getIDforPDF(pdf, pdfMaxDist, numBins, K, ambientDimension)

# Bradley you modified /home/xdaimon/.local/lib/python3.7/site-packages/scipy/optimize/minpack.py:795
# to silence a warning
