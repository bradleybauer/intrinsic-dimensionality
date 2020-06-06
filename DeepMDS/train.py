import torch as th
import numpy as np
import random
from model import DeepMDS
from loadData import loadImageNet


def criterion(dist_in, y1, y2, thresh=0.0):
    dist_out = th.norm(y1 - y2, dim=1, keepdim=True)
    diff = abs(dist_in - dist_out) - thresh
    ind_hard = diff > 0.0
    diff = (dist_in - dist_out)[ind_hard]
    if diff.size(0) != 0:
        return th.norm(diff) / diff.size(0)
    else:
        return None


def main():
    X = th.cuda.FloatTensor(loadImageNet())
    N = X.shape[0]

    layerSizes = (512, 256, 128, 64, 32)
    numLayers = len(layerSizes)
    epochs = 100
    batchSize = 128
    numBatches = N // batchSize
    lr = .5 * .001

    model = DeepMDS(layerSizes)
    model.load_state_dict(th.load('weights.pt'))
    print(model)
    model.to('cuda')
    optimizer = th.optim.Adam(model.parameters(), lr=lr)

    for layersToTrain in range(1, numLayers + 1):
        print()
        print('Training layers less than:', layersToTrain)
        for epoch in range(1, epochs + 1):
            lossSum = 0.
            print()
            print('epoch:', epoch)
            X = X[th.randperm(N)]
            for b in range(numBatches):
                x = X[b * batchSize:(b + 1) * batchSize]

                activations = model(x, layersToTrain)

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

                if loss != 0.0:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                lossSum += float(loss)

            print(lossSum / numBatches)

            print()
            print('dists in:', float(dist_in1[0]), 'dists out:', float(th.norm(y[0] - y[batchSize // 2])))

        th.save(model.state_dict(), 'new_weights.pt')


if __name__ == '__main__':
    main()
