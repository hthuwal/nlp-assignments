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
from sklearn.model_selection import StratifiedKFold
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


def get_time_dif(start_time):
    """
    Return the time used since start_time.
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def test(model, test_data):
    """
    Test the model on test dataset.
    """
    model.eval()
    print("\nTesting...")
    start_time = time.time()

    x, y = zip(*test_data)
    x = torch.from_numpy(np.array(x))
    y = torch.from_numpy(np.array(y))

    dataset = torch.utils.data.TensorDataset(x, y)
    test_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    y_true, y_pred = [], []
    for data, label in tqdm(test_loader):
        data, label = Variable(data, volatile=True), Variable(label, volatile=True)
        if use_cuda:
            data, label = data.cuda(), label.cuda()

        output = model(data)
        pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(label.data)

    test_acc = metrics.accuracy_score(y_true, y_pred)
    test_f1 = metrics.f1_score(y_true, y_pred, average='macro')
    print("\n\nTest accuracy: {0:>7.2%}, F1-Score: {1:>7.2%}".format(test_acc, test_f1))

    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_true, y_pred, target_names=['0', '1', '2']))

    print('Confusion Matrix...')
    cm = metrics.confusion_matrix(y_true, y_pred)
    print(cm)

    print("Time usage:", get_time_dif(start_time))
    return test_acc, test_f1


def train(model, train_data, num_epochs):

    x, y = zip(*train_data)
    x = torch.from_numpy(np.array(x))
    y = torch.from_numpy(np.array(y))

    # optimizer and loss function
    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # set the mode to train
    print("\nTraining")
    dataset = torch.utils.data.TensorDataset(x, y)
    train_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(num_epochs):
        print("\nEPOCH: %d" % epoch)
        # load the training data in batch
        model.train()
        for x_batch, y_batch in tqdm(train_loader):
            inputs, targets = Variable(x_batch), Variable(y_batch)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)  # forward computation
            loss = criterion(outputs, targets)

            # backward propagation and update parameters
            loss.backward()
            optimizer.step()

    return model


def doCrossValidation(x, y, model_file, fold=10, num_epochs=2):
    acc = []
    fscores = []
    iter = 1
    x = np.array(x)
    y = np.array(y)
    skf = StratifiedKFold(n_splits=fold)

    for train_index, test_index in skf.split(x, y):
        train_data = list(zip(x[train_index], y[train_index]))
        test_data = list(zip(x[test_index], y[test_index]))
        print("\nIter %d\n" % iter)
        iter += 1
        model = CNN()
        if os.path.exists(model_file):
            print("Loading Model")
            model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

        if use_cuda:
            model.cuda()

        model = train(model, train_data, num_epochs)
        curr_acc, curr_fscore = test(model, test_data)
        acc.append(curr_acc)
        fscores.append(curr_fscore)

    avg_acc = np.average(acc)
    avg_fscores = np.average(fscores)
    msg = "%d fold cross Validatioin\n Average Accuracy: %g\n Average Fscore: %g\n"
    print(msg % (fold, avg_acc, avg_fscores))


cv = True
x, y = load_data(data_file)
if not cv:
    model = CNN()
    if os.path.exists(model_file):
        print("Loading Model")
        model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    if use_cuda:
        model.cuda()

    test(model, zip(x, y))

else:
    doCrossValidation(x, y, model_file, num_epochs=10)
