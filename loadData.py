from torchvision.models import resnet34
from facenet_pytorch import InceptionResnetV1 as faceNet
from facenet_pytorch import MTCNN
from CharRNN.CharRNN import CharRNN
from CharRNN.helpers import charTensor

import torch as th
import numpy as np
import torchvision as tv
from PIL import Image
from multiprocessing import Process, Queue, Pool
import matplotlib.pyplot as plt
from tqdm import tqdm

import random
import unidecode
import os

th.backends.cudnn.benchmark = True
th.backends.cudnn.enabled = True


def loadMNIST(digit, fashion=False):
    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x.numpy().ravel())
    ])

    if not fashion:
        # Download training data
        train = tv.Datasets.MNIST(root='Datasets/', train=True, download=True, transform=transform)
        # Download testing data
        test = tv.Datasets.MNIST(root='Datasets/', train=False, download=True, transform=transform)
    else:
        # Download training data
        train = tv.Datasets.FashionMNIST(root='Datasets/', train=True, download=True, transform=transform)
        # Download testing data
        test = tv.Datasets.FashionMNIST(root='Datasets/', train=False, download=True, transform=transform)

    # Take only a single digit
    idx = train.targets == digit
    train.targets = train.targets[idx]
    train.data = train.data[idx]
    idx = test.targets == digit
    test.targets = test.targets[idx]
    test.data = test.data[idx]
    dataset = test

    dataset = th.utils.data.ConcatDataset([train, test])
    dataset = np.array(list(map(lambda x: x[0], dataset)))  # discard the label

    # vis = np.reshape(dataset[0], (28, 28))
    # plt.matshow(vis)
    # plt.show()

    return dataset


def loadFashionMNIST(digit):
    return loadMNIST(digit, fashion=True)


def loadLFW():
    embeddingFile = 'Datasets/lfw/lfwEmbeddings.npy'
    if os.path.exists(embeddingFile):
        dataset = np.load(embeddingFile)
        np.random.shuffle(dataset)
    else:
        print('\tEmbedding Images')

        # the code i use to embed lfw images has a memory leak somewhere
        # and so i have to do the embedding in a separate process to avoid oom.
        def processSubset(filePathBatches, queue):
            mtcnn = MTCNN()
            resnet = faceNet(pretrained='vggface2', device=th.device('cuda')).eval()
            embeddings = []
            for filePathBatch in filePathBatches:
                batch = [Image.open(f) for f in filePathBatch]
                cropWhitenedBatch = th.cat([t.unsqueeze(0) for t in mtcnn(batch)]).to('cuda')
                embeddings.append(resnet(cropWhitenedBatch).cpu().numpy())
            queue.put(embeddings)

        with th.no_grad():
            filePaths = []
            for root, dirs, files in os.walk('Datasets/lfw/lfw'):
                filePaths.extend(root + '/' + f for f in files if f.endswith('.jpg'))
            batchSize = 64
            filePathBatches = [filePaths[i:i + batchSize] for i in range(0, len(filePaths), batchSize)]
            numBatchesInSubset = 20
            subsets = [filePathBatches[i:i + numBatchesInSubset] for i in
                       range(0, len(filePathBatches), numBatchesInSubset)]
            embeddings = []
            for subset in tqdm(subsets):
                q = Queue()
                p = Process(target=processSubset, args=(subset, q))
                p.start()
                embeddings.extend(q.get())
                p.join()
            dataset = np.concatenate(embeddings)
            np.save(embeddingFile, dataset)
    return dataset


def loadImageNet(loadLabels=False):
    embeddingFile = 'Datasets/imagenet/imageNetEmbeddings.npy'
    labelsFile = 'Datasets/imagenet/imageNetLabels.npy'
    if os.path.exists(embeddingFile):
        dataset = np.load(embeddingFile)
        labels = np.load(labelsFile)
        np.random.shuffle(dataset)
    else:  # generate embeddings from images
        with th.no_grad():
            normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
            dataset = th.utils.data.DataLoader(
                tv.Datasets.ImageFolder('Datasets/imagenet/imagenet',
                                        tv.transforms.Compose([
                                            tv.transforms.RandomResizedCrop(224),
                                            tv.transforms.ToTensor(),
                                            normalize,
                                        ])),
                batch_size=64, shuffle=True, pin_memory=True)
            resnet = resnet34(pretrained=True, progress=True)
            classifier = list(resnet.children())[-1].eval().to('cuda')
            extractor = th.nn.Sequential(*list(resnet.children())[:-1]).eval().to('cuda')
            embeddings = []
            labels = []
            for xs, ys in tqdm(dataset):
                embedding = extractor(xs.to('cuda')).squeeze(2).squeeze(2)
                label = np.argmax(classifier(embedding).cpu().numpy(),axis=1)
                embedding = embedding.cpu().numpy()
                embeddings.append(embedding)
                labels.append(label)
            dataset = np.concatenate(embeddings)
            labels = np.concatenate(labels)
            np.save(embeddingFile, dataset)
            np.save(labelsFile, labels)
    if loadLabels:
        return dataset,labels
    return dataset


# Maybe use Guassian,Cube,Cauchy
def loadSynthetic(whichSet):
    sets = ['Sinusoid', 'S', 'Gauss', 'Moebius', 'M12']
    intrinsicDims = [[1], [3, 5, 7, 9], [3, 4, 5, 6], [2], [12]]
    ambientDims = [[3], [4, 6, 8, 10], [3, 4, 5, 6], [3], [72]]
    subsetSizes = [[400, 500, 600], [600, 800, 1000, 1200], [100, 200, 400, 800], [20, 40, 80, 120],
                   [200, 400, 800, 1600]]
    setName = sets[whichSet]
    # Each file contains a dataset partitioned into 90 subsets
    numSubsets = 90
    # for intrinsicDim, ambientDim in list(zip(intrinsicDims[whichSet], ambientDims[whichSet])):
    # for subsetSize in subsetSizes[whichSet][::-1]:
    ambientDim = ambientDims[whichSet][-1]
    intrinsicDim = intrinsicDims[whichSet][-1]
    subsetSize = subsetSizes[whichSet][-1]
    print('\t\tName:', setName)
    print('\t\tAmbientDim:', ambientDim)
    print('\t\tIntrinsicDim:', intrinsicDim)
    print('\t\tSubsetSize:', subsetSize)
    print('\t\tSubsetSize*NumSubsets:', subsetSize * numSubsets)

    randomSubset = random.randint(0, numSubsets - 1)
    # randomSubset = 10
    if len(ambientDims[whichSet]) > 1:
        filename = sets[whichSet] + str(intrinsicDim) + '_' + str(subsetSize) + '.BIN'
    else:
        filename = sets[whichSet] + '_' + str(subsetSize) + '.BIN'
    with open('Datasets/synthetic/data/' + filename, 'rb') as f:
        X = np.fromfile(f, dtype=np.float32)

        # https://stackoverflow.com/questions/23992956/row-wise-writing-to-file-using-fwrite-in-matlab
        # Matlab's fwrite saves a matrix in column major order.
        # The matrix that was written had shape (ambientDim,numSubsets*subsetSize)
        X = np.reshape(X, (numSubsets * subsetSize, ambientDim)).T

        X = X[:, subsetSize * randomSubset:subsetSize * (randomSubset + 1)]
        dataset = np.ascontiguousarray(X.T)
    print()
    return setName, dataset


def loadWiki():
    embeddingFile = 'Datasets/wiki103/wikiEmbeddings.npy'
    if os.path.exists(embeddingFile):
        dataset = np.load(embeddingFile)
    else:
        chunk_len = 200
        batch_size = 100
        n_datapoints = int(1e5)
        n_steps = int(n_datapoints / batch_size)
        n_hidden_to_keep = 10
        modelPath = 'Datasets/wiki103/wikiEmbeddingModel.pt'
        dataPath = 'Datasets/wiki103/wiki.train.raw'

        def randomDataPoint(chunk_len, batch_size, file, file_len):
            inp = th.LongTensor(batch_size, chunk_len)
            for bi in range(batch_size):
                start_index = random.randint(0, file_len - chunk_len)
                end_index = start_index + chunk_len + 1
                chunk = file[start_index:end_index]
                inp[bi] = charTensor(chunk[:-1])
            return inp.cuda()

        file = unidecode.unidecode(open(dataPath).read())
        file_len = len(file)
        with th.no_grad():
            decoder = CharRNN()
            decoder.load_state_dict(th.load(modelPath))
            decoder.cuda()
            embeddings = []
            for i in tqdm(range(n_steps)):
                data = randomDataPoint(chunk_len, batch_size, file, file_len)
                embeddings.append(decoder.embed(data, n_hidden_to_keep))
            dataset = np.concatenate(embeddings, axis=1)
            np.save(embeddingFile, dataset)

    return dataset


def loadDatasets():
    print("Loading Datasets!")
    datasets = {}

    # print('\tLoading MNIST-twos')
    # datasets['mnist-twos'] = loadMNIST(digit=2)
    #
    # print('\tLoading FashionMNIST-shirts')
    # datasets['fashion-shirts'] = loadFashionMNIST(digit=2)
    #
    # print('\tLoading LFW')
    # datasets['lfw'] = loadLFW()

    print('\tLoading ImageNet')
    datasets['imgnet'] = loadImageNet()

    # print('\tLoading Wiki')
    # data = loadWiki()
    # for t in range(0, 10):
    #     x = data[t, :, :]
    #     np.random.shuffle(x)
    #     datasets['wiki_t' + str(t + 1)] = x

    # print('\tLoading Synthetics')
    # for i in range(0, 4 + 1):
    #     name, data = loadSynthetic(i)
    #     if name == 'S':
    #         name = 'HyperSphere'
    #     datasets['synth_' + name] = data

    print("Done Loading datasets!")
    print()
    print()
    return datasets


if __name__ == '__main__':
    datasets = loadDatasets()
