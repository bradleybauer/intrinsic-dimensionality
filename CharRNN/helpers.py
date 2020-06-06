import unidecode
import string
import random
import time
import math
import torch

# Reading and un-unicode-encoding data

all_characters = string.printable
n_characters = len(all_characters)


def readFile(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)


# Turning a string into a tensor

def charTensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor


def charTensor(string):
    tensor = th.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = CharRNN.all_characters.index(string[c])
        except:
            continue
    return tensor
