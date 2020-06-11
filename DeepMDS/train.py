from tqdm import tqdm
import torch as th
import torch.nn as nn
import numpy as np
import sys
import random
import pickle
import os
from .model import DeepMDS
from loadData import loadImageNetEmbeddings
import torchvision as tv
from torchvision.models import resnet34


def criterion(dist_in, y1, y2, thresh=0.1):
    dist_out = th.norm(y1 - y2, dim=1, keepdim=True)
    diff = abs(dist_in - dist_out) - thresh
    ind_hard = diff > 0.0
    diff2 = (dist_in - dist_out)[ind_hard]
    # n = diff2.shape[0] - diff.shape[0]
    # if n:
    #     print(-n)
    if diff2.size(0) != 0:
        return th.norm(diff2) / diff2.size(0)
    else:
        return 0.


def getDataLoader(bs):
    batchSizeImg = bs
    normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return th.utils.data.DataLoader(
        tv.datasets.ImageFolder('Datasets/imagenet/imagenet',
                                tv.transforms.Compose([
                                    tv.transforms.RandomResizedCrop(224),
                                    tv.transforms.ToTensor(),
                                    normalize,
                                ])),
        batch_size=batchSizeImg, shuffle=True, pin_memory=True, num_workers=8)


def main():
    resnet = resnet34(pretrained=True, progress=True)
    extractor = nn.Sequential(*list(resnet.children())[:-1]).eval().cuda()
    resnet = list(resnet.children())[-1].eval().cuda()

    # Load an existing DeepMDS model or train a new one.
    if os.path.exists('DeepMDS/weights.pt'):
        with open('DeepMDS/layerSizes.pkl', 'rb') as f:
            layerSizes = pickle.load(f)
        deepMDS = DeepMDS(layerSizes)
        deepMDS.load_state_dict(th.load('DeepMDS/weights.pt'))
        deepMDS = deepMDS.cuda()
    else:
        layerSizes = (512, 256, 128, 64, 32)
        with open('DeepMDS/layerSizes.pkl', 'wb') as f:
            pickle.dump(layerSizes, f)
        deepMDS = DeepMDS(layerSizes)
        deepMDS = deepMDS.train().cuda()

        numLayers = len(layerSizes)
        epochs = 10
        batchSize = 1024
        lr = .005
        optimizer = th.optim.Adam(deepMDS.parameters(), lr=lr)

        while True:
            X = getDataLoader(512)
            embeddings = []
            with th.no_grad():
                for x, _ in tqdm(X):
                    x = x.cuda()
                    embeddings.append(extractor(x).squeeze(2).squeeze(2))
            X = th.cat(embeddings)
            # embeddings, _ = loadImageNetEmbeddings()
            # X = th.cuda.FloatTensor(embeddings).cuda()

            N = len(X)
            numBatches = N // batchSize
            for layersToTrain in range(1, numLayers + 1):
                print()
                print('Training layers less than:', layersToTrain)
                for epoch in range(1, epochs + 1):
                    lossSum = 0.
                    print('epoch:', epoch, '\tloss:', end=' ')
                    X = X[th.randperm(N)]
                    for b in range(numBatches):
                        x = X[b * batchSize:(b + 1) * batchSize]

                        activations = deepMDS.trainForward(x, layersToTrain)

                        dist_in1 = th.norm(x[:batchSize // 2] - x[batchSize // 2:], dim=1, keepdim=True)
                        dist_in2 = th.norm(x[0::2] - x[1::2], dim=1, keepdim=True)

                        losses = []
                        for y in activations:
                            crit = criterion(dist_in1, y[:batchSize // 2], y[batchSize // 2:])
                            if crit:
                                losses += [crit]
                            crit = criterion(dist_in2, y[0::2], y[1::2])
                            if crit:
                                losses += [crit]

                        for py, y in zip(activations, activations[1:]):
                            din = th.norm(py[:batchSize // 2] - py[batchSize // 2:], dim=1, keepdim=True)
                            crit = criterion(din, y[:batchSize // 2], y[batchSize // 2:])
                            if crit:
                                losses += [crit]
                            din = th.norm(py[0::2] - py[1::2], dim=1, keepdim=True)
                            crit = criterion(din, y[0::2], y[1::2])
                            if crit:
                                losses += [crit]

                        loss = 0.
                        if len(losses):
                            loss = sum(losses) / len(losses)

                        if loss:
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        lossSum += float(loss)

                    print(lossSum / numBatches)
            th.save(deepMDS.state_dict(), 'DeepMDS/weights.pt')

    classifier = nn.Linear(layerSizes[-1], 1000).cuda()
    # either train both mds and a linear layer
    net = nn.Sequential(deepMDS, classifier).cuda()
    # or just a new linear layer
    # net = classifier

    if os.path.exists('DeepMDS/classifierWeights.pt'):
        net.load_state_dict(th.load('DeepMDS/classifierWeights.pt'))

    lossF = nn.CrossEntropyLoss()
    # lr = .00987654321
    lr = .00287654321
    optimizer = th.optim.Adam(net.parameters(), lr)
    epochs = 20

    batchSize = 256
    while True:
        X = getDataLoader(batchSize)
        embeddings = []
        with th.no_grad():
            for x, _ in tqdm(X):
                x = x.cuda()
                embeddings.append(extractor(x).squeeze(2).squeeze(2))
            X = th.cat(embeddings)
            del x, embeddings
        #     embeddings, _ = loadImageNetEmbeddings()
        #     X = th.cuda.FloatTensor(embeddings).cuda()
        numBatches = len(X) // batchSize
        for epoch in range(1, epochs):
            avg_loss = 0.
            trainacc = 0.
            N = X.shape[0]
            X = X[th.randperm(N)]
            for b in range(numBatches):
                with th.no_grad():
                    emb = X[b*batchSize:(b+1)*batchSize]
                    y = resnet(emb).argmax(axis=1)
                # forward
                # yhat = net(deepMDS(emb))
                yhat = net(emb)
                # compute error
                loss = lossF(yhat, y)
                avg_loss += float(loss)
                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                trainacc += float((yhat.argmax(axis=1) == y).sum()) / float(batchSize)
            print('epoch', epoch)
            print('TrainAcc:', trainacc/numBatches)
            th.save(net.state_dict(), 'DeepMDS/classifierWeights.pt')
        del loss, emb, y, yhat


if __name__ == '__main__':
    main()
