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

# hyperparameters
EPOCH = 2
BATCH_SIZE = 1000
LR = 0.001
DOWNLOAD_MNIST = False


class CNN(nn.Module):
    def __init__(self,):
        super(CNN, self).__init__()
        self.input_droput = nn.Dropout(p=0.2)
        self.conv1 = nn.Sequential(  # 1 x 300
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.2),
        )  # 16 x 150
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 8, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=0.1),
        )  # 8 x 75
        self.conv3 = nn.Sequential(
            nn.Conv1d(8, 4, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool1d(5),
        )  # 4 x 15
        self.out = nn.Linear(4 * 15, 5)

    def forward(self, x):
        # print("\ndropout")
        x = self.input_droput(x)
        # print("conv1")
        x = self.conv1(x)
        # print("conv2")
        x = self.conv2(x)
        # print("conv3")
        x = self.conv3(x)
        # print("output")
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


print("Loading Doc2vec model")
model = gensim.models.Doc2Vec.load("models/doc2vec")
x_train = torch.from_numpy(model.docvecs.vectors_docs)

print("\nReading y_train\n")
y_train = []
with open("../dataset/audio_train.json", "r") as f:
    for line in tqdm(f):
        temp = json.loads(line)
        y_train.append(int(temp["overall"]-1))

y_train = torch.from_numpy(np.array(y_train))


print("Loading Doc2vec of dev data")
x_dev, y_dev = pickle.load(open("dev_data.doc2vec", "rb"))
x_dev = torch.from_numpy(np.array(x_dev))
x_dev = x_dev.view(x_dev.size(0), 1, x_dev.size(1))
y_dev = [ int(y) - 1 for y in y_dev ]
y_dev = torch.from_numpy(np.array(y_dev))

# print("\nReadin dev_data\n")
# x_dev = []
# y_dev = []
# with open("../dataset/audio_dev.json", "r") as f:
#     for line in tqdm(f):
#         temp = json.loads(line)
#         x_dev.append((temp["summary"] + " ") * 4 + temp["reviewText"])
#         y_dev.append(temp["overall"])

# print("Converting test_data to doc2vec")
# x_dev = [ model.infer_vector(each.split(), alpha=0.025, min_alpha=0.025) for each in tqdm(x_dev)]
# pickle.dump((x_dev, y_dev), open("dev_data.doc2vec", "wb"))

if os.path.exists("models/cnn"):
    print("Loading Model")
    cnn, optimizer, loss_func = torch.load("models/cnn")
else:
    cnn = CNN()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCH):
    print("EPOCH: %d" %(epoch))
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x)
        b_x = b_x.view(b_x.size(0), 1, b_x.size(1))
        b_y = Variable(y)
        # b_y = b_y.view(b_y.size(0))

        # print("Output by CNN")
        output = cnn(b_x) # output of the cnn
        # print(output.shape)
        # print("Calculating loss")
        loss = loss_func(output, b_y) # loss
        # print("Claering old gradients")
        optimizer.zero_grad() # clearing gradients
        # print("Backpropogating")
        loss.backward() # backpropogation
        # print("applygradients")
        optimizer.step() # applygradients
    
        if step % 80 == 0:
            test_output = cnn(x_dev)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            # print(pred_y[1:10])
            # print(y_dev[1:10])
            accuracy = sum(pred_y == y_dev) / float(y_dev.size(0))
            f_score = f1_score(y_dev.numpy(), pred_y.numpy(), average="macro")
            print('Epoch: ', epoch, 'step: ', step, '| train loss: %.4f' % loss.data[0], '| dev accuracy: %.2f' % accuracy, '| f score: %.2f' % f_score)

    torch.save((cnn, optimizer, loss_func), "models/cnn")