from Estimators.mst import idMst, idMstExact, knnMstWeight, euclideanMstWeight
from Estimators.graph import idGraph, getKnnRelation, getPDF
from Estimators.fisher import idFisher
from Estimators.idea import idIDEA

import numpy as np
import torch as th
import pickle
import os

from deeprobust.image.attack.deepfool import DeepFool
# from deeprobust.image.attack.pgd import PGD
from DeepMDS.model import DeepMDS

import torchvision as tv
import torch.nn as nn
from torchvision.models import resnet34

th.backends.cudnn.benchmark = True
th.backends.cudnn.enabled = True


def testDistancePdfEstimator(datasets):
    K = 12

    print('Testing Distance PDF Estimator')
    for name, data in datasets.items():
        print('Dataset:', name)

        numberOfNodes = data.shape[0]
        numBins = max(min(numberOfNodes // 25, 1000), 100)
        numSamplesSmall = 25
        numSamplesBig = min(300, len(data))

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
        print('Testing estimators on', name, '. |X|=', len(dataset))

        print('\tIDEA: ', end='')
        estGraph = idIDEA(dataset, K=20)
        print(estGraph)

        if len(dataset) < 5000:
            print('\tMST exact: ', end='')
            exactMst = idMstExact(dataset)
            print(exactMst)

        print('\tMST approx: ', end='')
        estMst = idMst(dataset)
        print(estMst)

        print('\tGraph: ', end='')
        estGraph = idGraph(dataset)
        print(estGraph)

        if len(dataset) < 5000:
            print('\tFisher: ', end='')
            estFisher = idFisher(dataset)
            print(estFisher)

        print()
        print()


def testAdvAttackImageNet():
    normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = th.utils.data.DataLoader(
        tv.datasets.ImageFolder('Datasets/imagenet/imagenet',
                                tv.transforms.Compose([
                                    tv.transforms.RandomResizedCrop(224),
                                    tv.transforms.ToTensor(),
                                    normalize,
                                ])),
        batch_size=1, shuffle=True, pin_memory=True, num_workers=8)

    resnet = resnet34(pretrained=True, progress=True).eval().cuda()
    perturb = DeepFool(resnet)
    # perturb = PGD(resnet)

    extractor = th.nn.Sequential(*list(resnet.children())[:-1]).eval().cuda()
    with open('DeepMDS/layerSizes.pkl', 'rb') as f:
        layerSizes = pickle.load(f)
    deepMDS = DeepMDS(layerSizes)
    deepMDS = deepMDS.eval().cuda()
    classifier = nn.Linear(layerSizes[-1], 1000).eval().cuda()
    net = nn.Sequential(deepMDS, classifier).eval().cuda()
    if os.path.exists('DeepMDS/classifierWeights.pt'):
        net.load_state_dict(th.load('DeepMDS/classifierWeights.pt'))
    perturbMDS = DeepFool(nn.Sequential(extractor, net).eval().cuda())
    # perturbMDS = PGD(nn.Sequential(extractor, net).eval().cuda())

    avg_r_norm = 0.
    avg_iters = 0.
    mds_avg_r_norm = 0.
    mds_avg_iters = 0.
    n = 1
    acc = 0
    mitersRes = 0
    mitersMDS = 0
    for i, (x, _) in enumerate(dataset):
        x = x.cuda()
        with th.no_grad():
            resnetClass = resnet(x).argmax(axis=1)
            mdsClass = net(extractor(x)).argmax(axis=1)
        if mdsClass == resnetClass:
            acc += 1
            print('Acc', acc / (i + 1))

            # Test resnet
            # use pgd
            # r = (perturb.generate(x[:], mdsClass) - x[:]).cpu().detach()
            # or use DeepFool
            perturb.generate(x[:], mdsClass)
            r, iters = perturb.getpert()
            avg_iters += iters
            if iters > mitersRes:
                mitersRes = iters

            avg_r_norm += np.linalg.norm(r)
            print('Normal:', avg_r_norm / n, '\t', avg_iters / n, '\t', mitersRes)

            # Test resnet + DeepMDS
            # use pgd
            # r = (perturbMDS.generate(x[:], mdsClass) - x[:]).cpu().detach()
            # or use DeepFool
            perturbMDS.generate(x[:], mdsClass)
            r, iters = perturbMDS.getpert()
            mds_avg_iters += iters
            if iters > mitersMDS:
                mitersMDS = iters

            mds_avg_r_norm += np.linalg.norm(r)
            print('MDS:', mds_avg_r_norm / n, '\t', mds_avg_iters / n, '\t', mitersMDS)
            n += 1
            print()
            print()

        # Check that the classification is different (optional)
        # y = classifier(extractor(x.cuda()).squeeze()).argmax()
        # perturb.generate(x, y)
        # x = x+th.from_numpy(r)
        # y = classifier(extractor(x.cuda()).squeeze()).argmax()

    # Gradient feature extraction feature representation backward pass ??? hmm bradley... try.. to.. remember....
