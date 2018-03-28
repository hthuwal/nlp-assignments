import json
import numpy as np
import os
import pickle
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
BATCH_SIZE = 1
EPOCHS = 10

print("Lodaing Vocab")
word2idx = pickle.load(open("word2idx", "rb"))
V = len(word2idx)

max_length = 300
embed_size = 128
model_file = "hc_4.model"

data_file = "../dataset/data/labelled.json"


class CNN(nn.Module):
    """
    CNN text classification model, based on the paper.
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(V + 2, embed_size)  # embedding layer

        # three different convolutional layers
        Ks = [3, 4, 5]
        num_filters = 100
        num_classes = 3
        self.convs = nn.ModuleList([nn.Conv1d(embed_size, num_filters, k) for k in Ks])
        self.dropout = nn.Dropout(0.5)  # a dropout layer
        self.fc1 = nn.Linear(3 * num_filters, num_classes)  # a dense layer for classification

    @staticmethod
    def conv_and_max_pool(x, conv):
        """Convolution and global max pooling layer"""
        return F.relu(conv(x).permute(0, 2, 1).max(1)[0])

    def forward(self, inputs):
        # Conv1d takes in (batch, channels, seq_len), but raw embedded is (batch, seq_len, channels)
        embedded = self.embedding(inputs).permute(0, 2, 1)
        x = [self.conv_and_max_pool(embedded, k) for k in self.convs]  # convolution and global max pooling
        x = self.fc1(self.dropout(torch.cat(x, 1)))  # concatenation and dropout

        return x


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


def cv_get_data(x, y, fold=10):
    z = list(zip(x, y))
    shuffle(z)
    fold_size = int(len(z) / fold)
    for i in range(0, len(z), fold_size):
        test = z[i:i + (fold_size)]
        train = z[0:i] + z[i + (fold_size):]
        yield (test, train)


