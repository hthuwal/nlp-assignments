import json
import os
import pickle
import sys
import gensim
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import numpy as np
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
import random
en_stop = set(stopwords.words('english'))

# hyperparameters
EPOCH = 2
BATCH_SIZE = 100
LR = 0.001
DOWNLOAD_MNIST = False


class CNN(nn.Module):
    def __init__(self,):
        super(CNN, self).__init__()
        self.embed = nn.Embedding(15002, 300)
        self.input_droput = nn.Dropout(p=0.2)
        self.conv1 = nn.Sequential(  # 1 x 300
            nn.Conv1d(
                in_channels=1500,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.2),
        )  # 64 x 150
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 32, 3, 1, 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=0.1),
        )  # 32 x 76
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 16, 3, 1, 2),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Dropout(p=0.1),
        )  # 16 x 26
        self.out = nn.Linear(16 * 26, 5)

    def forward(self, x):
        # print("Input to Embedding: ", x.size())
        x = self.embed(x)
        # print("Input to Dropout: ", x.size())
        x = self.input_droput(x)
        # print("Input to conv1: ", x.size())
        x = self.conv1(x)
        # print("Input to conv2: ", x.size())
        x = self.conv2(x)
        # print("Input to conv3: ", x.size())
        x = self.conv3(x)
        # print("Input to conv4: ", x.size())
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

print("Lodaing Vocab")
word2idx = pickle.load(open("word2idx", "rb"))

print("\nReading Training Data\n")
data = []
with open("../dataset/audio_train.json", "r") as f:
    for line in tqdm(f):
        data.append(json.loads(line))

print("\nExtracting x's and y's\n")
corpus = [(each["summary"]+" ")*4 + each["reviewText"] for each in data]
y_train = np.array([int(each["overall"]-1) for each in data])

del data
print("\nRemoving stopwords\n")
x_train = []
for each in tqdm(corpus):
    temp = each.lower().split()
    x_train.append([ word2idx[i] for i in temp if i not in en_stop ])
del corpus

print("\nCoverting to sets\n")
for i in tqdm(range(len(x_train))):
    x_train[i] = list(set(x_train[i]))

max_length = 1500

for i in tqdm(range(len(x_train))):
    if len(x_train[i]) < max_length:
        x_train[i] = x_train[i] + [15001 for j in range(max_length - len(x_train[i]))]

x_train = torch.from_numpy(np.array(x_train))
y_train = torch.from_numpy(y_train)



print("\nReading Dev Data\n")
data = []
with open("../dataset/audio_dev.json", "r") as f:
    for line in tqdm(f):
        data.append(json.loads(line))

print("\nExtracting x's and y's\n")
corpus = [(each["summary"]+" ")*4 + each["reviewText"] for each in data]
y_dev = np.array([int(each["overall"]-1) for each in data])

del data
print("\nRemoving stopwords\n")
x_dev = []
for each in tqdm(corpus):
    temp = each.lower().split()
    x_dev.append([ word2idx[i] for i in temp if i not in en_stop ])
del corpus

for i in tqdm(range(len(x_dev))):
    x_dev[i] = list(set(x_dev[i]))

for i in range(len(x_dev)):
    if len(x_dev[i]) < max_length:
        x_dev[i] = x_dev[i] + [15001 for j in range(max_length - len(x_dev[i]))]

data = random.sample(list(zip(x_dev, y_dev)), 1000)
x_dev, y_dev = zip(*data)
x_dev = torch.from_numpy(np.array(x_dev))
y_dev = torch.from_numpy(np.array(y_dev))



if os.path.exists("models/cnn_we"):
    print("Loading Model")
    cnn, optimizer, loss_func = torch.load("models/cnn")
else:
    cnn = CNN()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

dev_dataset = torch.utils.data.TensorDataset(x_dev, y_dev)
dev_loader = Data.DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCH):
    print("EPOCH: %d" % (epoch))
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x)
        b_x = b_x.view(b_x.size(0), b_x.size(1))
        b_y = Variable(y)
        # b_y = b_y.view(b_y.size(0))

        # print("Output by CNN")
        output = cnn(b_x)  # output of the cnn
        # print(output.shape)
        # print("Calculating loss")
        loss = loss_func(output, b_y)  # loss
        # print("Claering old gradients")
        optimizer.zero_grad()  # clearing gradients
        # print("Backpropogating")
        loss.backward()  # backpropogation
        # print("applygradients")
        optimizer.step()  # applygradients

        if step % 80 == 0:
            cnn.eval()
            d_x = Variable(x_dev)
            d_x = d_x.view(d_x.size(0), d_x.size(1))
            d_y = Variable(y_dev)
            test_output = cnn(d_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            
            accuracy = sum(pred_y == y_dev) / float(y_dev.size(0))
            f_score = f1_score(y_dev.numpy(), pred_y.numpy(), average="macro")
            print('Epoch: ', epoch, 'step: ', step, '| train loss: %.4f' % loss.data[0], '| dev accuracy: %.2f' % accuracy, '| f score: %.2f' % f_score)

        cnn.train()
    torch.save((cnn, optimizer, loss_func), "models/cnn_we")
