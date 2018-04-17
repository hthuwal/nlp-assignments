import os
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn import metrics
from tqdm import tqdm

use_cuda = torch.cuda.is_available()


def read_data(file):
    data = []
    labels = []
    with open(file, "r", errors='replace') as f:
        temp_data = []
        temp_labels = []
        for line in f:
            if line.strip() == "":
                data.append(temp_data)
                labels.append(temp_labels)
                temp_data = []
                temp_labels = []
            else:
                temp_data.append(line.strip().split()[0])
                temp_labels.append(line.strip().split()[1])

    return data, labels


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, label_size, batch_size=1):
        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.hd2labels = nn.Linear(2 * hidden_dim, label_size)
        self.hidden = self.init_hidden(batch_size)

    def init_hidden(self, batch_size):
        a = torch.randn(2, batch_size, self.hidden_dim)
        b = torch.randn(2, batch_size, self.hidden_dim)
        if use_cuda:
            a = a.cuda()
            b = b.cuda()

        return Variable(a), Variable(b)

    def forward(self, data):
        embeddings = self.embeds(data).view(1, len(data), -1)
        # print(embeddings.size())
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        # print(lstm_out.size())
        lstm_out = lstm_out.view(len(data), -1)
        # print(lstm_out.size())
        output = self.hd2labels(lstm_out)
        # print(output.size())
        return output


def wlist2ilist(sentence, w2i):
    return [w2i[word] for word in sentence]


def get_w2i(data):
    w2i = {}
    for sentence in train_data:
        for word in sentence:
            if word not in w2i:
                w2i[word] = len(w2i) + 1
    return w2i


train_data, train_labels = read_data("train.txt")

w2i = get_w2i(train_data)
t2i = {"D": 0, "T": 1, "O": 2}
vocab_size = len(w2i) + 1

data = list(zip(train_data, train_labels))
random.shuffle(data)
train_data, train_labels = zip(*data)

dev_data, dev_labels = train_data[3001:], train_labels[3001:]
train_data, train_labels = train_data[:3001], train_labels[:3001]


def train(epochs=1000, file="bilstm.model"):
    model = BiLSTM(vocab_size, 100, 100, len(t2i))
    if os.path.exists(file):
        print("Loading Model")
        model.load_state_dict(torch.load(file, map_location=lambda storage, loc: storage))

    if use_cuda:
        model.cuda()

    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\nTraining...")
    model.train()
    best_macrof1 = 0
    for epoch in range(epochs):
        print("\nEpoch %d" % (epoch))
        i = 1
        for sentence, labels in zip(train_data, train_labels):
            print("\r%d/%d" % (i, len(train_labels)), end="")
            i += 1
            inp = Variable(torch.LongTensor(wlist2ilist(sentence, w2i)))
            out = Variable(torch.LongTensor(wlist2ilist(labels, t2i)))

            optimizer.zero_grad()
            # print("Forward...", end=" ")
            model.hidden = model.init_hidden(1)

            if use_cuda:
                inp = inp.cuda()
                out = out.cuda()

            outputs = model(inp)

            # print("Gradients...", end=" ")
            loss = criterion(outputs, out)
            # print("backwards...", end="\n")
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            model.eval()
            print("\nTesting...")
            y_pred = []
            y_true = []
            i = 1
            for sentence, labels in zip(dev_data, dev_labels):
                print("\r%d/%d" % (i, len(dev_labels)), end="")
                i += 1
                inp = Variable(torch.LongTensor(wlist2ilist(sentence, w2i)))
                out = Variable(torch.LongTensor(wlist2ilist(labels, t2i)))

                optimizer.zero_grad()
                model.hidden = model.init_hidden(1)

                if use_cuda:
                    inp = inp.cuda()
                    out = out.cuda()

                outputs = model(inp)
                pred = torch.max(outputs, dim=1)[1].data.cpu().numpy().tolist()

                y_pred.extend(pred)
                y_true.extend(out.data.cpu().numpy().tolist())

            macro_f1 = metrics.f1_score(y_true, y_pred, average='macro', labels=[0, 1])
            if macro_f1 > best_macrof1:            # store the best result
                best_macrof1 = macro_f1
                improved_str = '*'
                torch.save(model.state_dict(), file)
            else:
                improved_str = ""

            print("\nAccuracy: %f" % (metrics.accuracy_score(y_true, y_pred)))
            print("Macro F1: %f" % (macro_f1) + improved_str)
            print(metrics.classification_report(y_true, y_pred, labels=[0, 1]))
            model.train()
            print("Training...")


train()
