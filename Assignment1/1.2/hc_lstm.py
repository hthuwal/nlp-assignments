import json
import os
import pickle
import sys
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import numpy as np
from sklearn import metrics
from nltk.corpus import stopwords
import torch.nn.functional as F
import torch.autograd as autograd
# from google.colab import files
import random
from datetime import timedelta
en_stop = set(stopwords.words('english'))
use_cuda = torch.cuda.is_available()
BATCH_SIZE = 50
EPOCHS = 10
torch.set_num_threads(8)

print("Lodaing Vocab")
word2idx = pickle.load(open("word2idx", "rb"))
V = len(word2idx)

max_length = 300
embed_size = 128
model_file = "lstm.model"
print("Lodaing Vocab")
word2idx = pickle.load(open("word2idx", "rb"))
V = len(word2idx)

max_length = 300
embed_size = 128
model_file = "lstm.model"

print("\nReading Training Data\n")
data = []
with open("../dataset/audio_train.json", "r") as f:
    for line in tqdm(f):
        data.append(json.loads(line))

print("\nExtracting x's and y's\n")
corpus = [(each["summary"] + " ") * 4 + each["reviewText"] for each in data]
y_train = [int(each["overall"] - 1) for each in data]
y_train = [0 if i == 0 or i == 1 else 2 if i == 3 or i == 4 else 1 for i in y_train]

del data
print("\nRemoving stopwords\n")
x_train = []
for each in tqdm(corpus):
    temp = each.lower().split()
    x_train.append([word2idx[i] for i in temp if i not in en_stop])
del corpus

print("\nCoverting to sets\n")
for i in tqdm(range(len(x_train))):
    x_train[i] = list(set(x_train[i]))


for i in tqdm(range(len(x_train))):
    if len(x_train[i]) < max_length:
        x_train[i] = x_train[i] + [V+1 for j in range(max_length - len(x_train[i]))]
    else:
        x_train[i] = x_train[i][0:max_length]

print("\nReading Dev Data\n")
data = []
with open("../dataset/audio_dev.json", "r") as f:
    for line in tqdm(f):
        data.append(json.loads(line))

print("\nExtracting x's and y's\n")
corpus = [(each["summary"] + " ") * 4 + each["reviewText"] for each in data]
y_dev = [int(each["overall"] - 1) for each in data]
y_dev = [0 if i == 0 or i == 1 else 2 if i == 3 or i == 4 else 1 for i in y_dev]

del data
print("\nRemoving stopwords\n")
x_dev = []
for each in tqdm(corpus):
    temp = each.lower().split()
    x_dev.append([word2idx[i] for i in temp if i not in en_stop])
del corpus

print("\nCoverting to sets\n")
for i in tqdm(range(len(x_dev))):
    x_dev[i] = list(set(x_dev[i]))

for i in tqdm(range(len(x_dev))):
    if len(x_dev[i]) < max_length:
        x_dev[i] = x_dev[i] + [V+1 for j in range(max_length - len(x_dev[i]))]
    else:
        x_dev[i] = x_dev[i][0:max_length]

data = random.sample(list(zip(x_dev, y_dev)), 1000)
sample_x_dev, sample_y_dev = zip(*data)

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        print(embeds.size())
        x = embeds.view(max_length, embed_size, -1)
        print(x.size())
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)
        print(log_probs)
        return log_probs


def get_time_dif(start_time):
    """
    Return the time used since start_time.
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def test(model, x_list, y_list):
    """
    Test the model on test dataset.
    """
    print("Testing...")
    start_time = time.time()
    
    # restore the best parameters
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    y_true, y_pred = [], []
    for data, label in tqdm(zip(x_list, y_list)):
        data, label = Variable(torch.LongTensor(data), volatile=True), Variable(torch.LongTensor([label]), volatile=True)
        if use_cuda:
            data, label = data.cuda(), label.cuda()

        output = model(data)
        pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(label.data)

    test_acc = metrics.accuracy_score(y_true, y_pred)
    test_f1 = metrics.f1_score(y_true, y_pred, average='macro')
    print("Test accuracy: {0:>7.2%}, F1-Score: {1:>7.2%}".format(test_acc, test_f1))

    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_true, y_pred, target_names=['0', '1', '2']))

    print('Confusion Matrix...')
    cm = metrics.confusion_matrix(y_true, y_pred)
    print(cm)

    print("Time usage:", get_time_dif(start_time))

def evaluate(x_list, y_list, model, loss, data_len, verbose=False):
    """
    Evaluation, return accuracy and loss
    """
    model.eval()  # set mode to evaluation to disable dropout
    total_loss = 0.0
    y_true, y_pred = [], []

    count = 0
    for data, label in zip(x_list, y_list):
        count += 1
        if verbose:
           sys.stdout.write("\r\x1b[K %d/%d" % (count, len(x_list)))
           sys.stdout.flush()
        data, label = Variable(torch.LongTensor(data), volatile=True), Variable(torch.LongTensor([label]), volatile=True)
        if use_cuda:
            data, label = data.cuda(), label.cuda()

        output = model(data)
        losses = loss(output, label)

        total_loss += losses.data[0]
        pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(label.data)

    acc = (np.array(y_true) == np.array(y_pred)).sum()
    fscore = metrics.f1_score(y_true, y_pred, average="macro")
    return acc / data_len, total_loss / data_len, fscore
  
def train():

    start_time = time.time()
    model = LSTMClassifier(embed_size, 50, V+2, 3)
    print(model)

    if os.path.exists(model_file):
        print("Loading Model from %s" %model_file)
        model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
    if use_cuda:
        model.cuda()

    # optimizer and loss function
    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # set the mode to train
    print("Training and evaluating...")

    best_fscore = 0.0
    for epoch in range(EPOCHS):
        # load the training data in batch
        model.train()
        iter = 0
        for x_batch, y_batch in tqdm(zip(x_dev, y_dev)):
            inputs, targets = Variable(torch.LongTensor(x_dev)), Variable(torch.LongTensor([y_batch]))
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)  # forward computation
            loss = criterion(outputs, targets)

            # backward propagation and update parameters
            loss.backward()
            optimizer.step()
            if iter % 1000 == 0 and iter!=0:
                test_acc, test_loss, fscore = evaluate(sample_x_dev, sample_y_dev, criterion, len(sample_y_dev))
                print(iter, test_acc, test_loss, fscore)
                model.train()

            iter += 1
        # evaluate on both training and test dataset
        # train_acc, train_loss = evaluate(train_loader, model, criterion, len(y_train))
        test_acc, test_loss, fscore = evaluate(x_dev, y_dev, model, criterion, len(y_dev))
        

        if fscore > best_fscore:
            # store the best result
            best_fscore = fscore
            improved_str = '*'
            filename = model_file
            print("Saving model in %s" %model_file)
            torch.save(model.state_dict(), filename)
            
        else:
            improved_str = ''

        time_dif = get_time_dif(start_time)
        msg = "Epoch {0:3}, " \
              + "Test_loss: {1:>6.2}, Test_acc {2:>6.2%}, Fscore {3:6.2%} Time: {4} {5}"
        print(msg.format(epoch + 1, test_loss, test_acc, fscore, time_dif, improved_str))
    test(model, dev_loader)

train()