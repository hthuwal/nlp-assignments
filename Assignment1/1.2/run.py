import json
import os
import pickle
import sys
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision


from torch.autograd import Variable
import numpy as np
from nltk.corpus import stopwords

en_stop = set(stopwords.words('english'))
use_cuda = torch.cuda.is_available()

print("Lodaing Vocab")
word2idx = pickle.load(open("2017MCS2074.vocab", "rb"))  # word2idx
V = len(word2idx)

model_file = "2017MCS2074.cnn"  # hc_15000_3_4_5.model
max_length = 300
BATCH_SIZE = 50
embed_size = 128
testfile = sys.argv[1]
outfile = sys.argv[2]

print("\nReading test Data:")
data = []
with open(testfile, "r") as f:
    count = 0
    for line in f:
        count += 1
        sys.stdout.write("\r%d" % (count))
        sys.stdout.flush()
        data.append(json.loads(line))

print("\nExtracting x's and y's")
corpus = [(each["summary"] + " ") * 4 + each["reviewText"] for each in data]
y_dev = [int(each["overall"] - 1) for each in data]
y_dev = [0 if i == 0 or i == 1 else 2 if i == 3 or i == 4 else 1 for i in y_dev]

del data

print("\nRemoving stopwords:")
x_dev = []
count = 0
for each in corpus:
    count += 1
    sys.stdout.write("\r" + "%d/%d" % (count, len(corpus)))
    sys.stdout.flush()
    temp = each.lower().split()
    x_dev.append([word2idx[i] for i in temp if i not in en_stop])

del corpus

print("\nCoverting to sets:")
count = 0
for i in range(len(x_dev)):
    count += 1
    sys.stdout.write("\r" + "%d/%d" % (count, len(x_dev)))
    sys.stdout.flush()
    x_dev[i] = list(set(x_dev[i]))

print("\nAdjusting Document Lengths: ")
count = 0
for i in range(len(x_dev)):
    count += 1
    sys.stdout.write("\r" + "%d/%d" % (count, len(x_dev)))
    sys.stdout.flush()
    if len(x_dev[i]) < max_length:
        x_dev[i] = x_dev[i] + [V + 1 for j in range(max_length - len(x_dev[i]))]
    else:
        x_dev[i] = x_dev[i][0:max_length]

x_dev = torch.from_numpy(np.array(x_dev))
y_dev = torch.from_numpy(np.array(y_dev))

dataset = torch.utils.data.TensorDataset(x_dev, y_dev)
dev_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE)

if use_cuda:
    x_dev = x_dev.cuda()
    y_dev = y_dev.cuda()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(V + 2, embed_size)  # embedding layer
        Ks = [3, 4, 5]
        num_filters = 100
        num_classes = 3
        self.convs = nn.ModuleList([nn.Conv1d(embed_size, num_filters, k) for k in Ks])
        self.dropout = nn.Dropout(0.5)  # a dropout layer
        self.fc1 = nn.Linear(3 * num_filters, num_classes)

    @staticmethod
    def conv_and_max_pool(x, conv):
        return F.relu(conv(x).permute(0, 2, 1).max(1)[0])

    def forward(self, inputs):
        embedded = self.embedding(inputs).permute(0, 2, 1)
        x = [self.conv_and_max_pool(embedded, k) for k in self.convs]  # convolution and global max pooling
        x = self.fc1(self.dropout(torch.cat(x, 1)))  # concatenation and dropout

        return x


def test(model, test_loader):
    print("\nPredicting: ")
    model.eval()
    y_true, y_pred = [], []
    count = 0
    for data, label in test_loader:
        count += 1
        sys.stdout.write("\r" + "Batch: %d/%d" % (count, len(test_loader)))
        sys.stdout.flush()
        data, label = Variable(data, volatile=True), Variable(label, volatile=True)
        if use_cuda:
            data, label = data.cuda(), label.cuda()

        output = model(data)
        pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(label.data)

    print("\nSaving results to %s" % (outfile))
    with open(outfile, "w") as f:
        y_pred = ["1" if i == 0 else "3" if i == 1 else "5" for i in y_pred]
        for each in y_pred:
            f.write("%s\n" % (each))


model = CNN()
if os.path.exists(model_file):
    print("\nLoading Model")
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
else:
    print("\nModel not found. Plese download model file: %s" % (model_file))

if use_cuda:
    model.cuda()

test(model, dev_loader)
odel, dev_loader)
