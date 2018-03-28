import json
import numpy as np
import os
import pickle
import random
import sys
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from nltk.corpus import stopwords
from torch.autograd import Variable
from tqdm import tqdm
from sklearn import metrics
from datetime import timedelta
from random import shuffle
en_stop = set(stopwords.words('english'))
use_cuda = torch.cuda.is_available()
BATCH_SIZE = 50
EPOCHS = 10

print("Lodaing Vocab")
word2idx = pickle.load(open("word2idx", "rb"))
V = len(word2idx)

max_length = 300
embed_size = 128
model_file = "hc_4.model"

train_file = "dataset/data/labelled.json"


def load_data(file):
    print("\nReading Training Data\n")
    data = []
    with open(file, "r") as f:
        for line in tqdm(f):
            data.append(json.loads(line))

    print("\nExtracting x's and y's\n")
    corpus = [(each["summary"] + " ") * 4 + each["reviewText"] for each in data]
    y = [int(each["overall"] - 1) if "overall" in each else -1 for each in data]
    y = [0 if i == 0 or i == 1 else 2 if i == 3 or i == 4 else 1 for i in y]

    del data
    print("\nRemoving stopwords\n")
    x = []
    for each in tqdm(corpus):
        temp = each.lower().split()
        x.append([word2idx[i] for i in temp if i not in en_stop])
    del corpus

    print("\nCoverting to sets\n")
    for i in tqdm(range(len(x))):
        x[i] = list(set(x[i]))

    for i in tqdm(range(len(x))):
        if len(x[i]) < max_length:
            x[i] = x[i] + [V + 1 for j in range(max_length - len(x[i]))]
        else:
            x[i] = x[i][0:max_length]

    return x, y


def crossvalidation(x, y, fold=10):
    z = zip(x, y)
