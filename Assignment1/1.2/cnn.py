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

# hyperparameters
EPOCH = 1
BATCH_SIZE = 1000
LR = 0.001
DOWNLOAD_MNIST = False


class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.input_droput = nn.Dropout(p=0.2)
		self.conv1 = nn.Sequential(  # 300 x 1
			nn.Conv1d(
				in_channels=1,
				out_channels=64,
				kernel_size=5,
				stride=1,
				padding=2,
			),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2),
			nn.Dropout(p=0.2),
		)  # 64 x 150 x 150
		self.conv2 = nn.Sequential(
			nn.Conv1d(64, 32, 5, 1, 2),
			nn.ReLU(),
			nn.MaxPool1d(2),
			nn.Dropout(p=0.1),
		)  # 32 x 75 x 75
		self.conv3 = nn.Sequential(
			nn.Conv1d(32, 16, 5, 1, 2),
			nn.ReLU(),
			nn.MaxPool1d(3),
		)  # 16 x 15 x 15
		self.out = nn.Linear(16 * 15 * 15, 5)

	def forward(self, x):
		x = self.input_droput(x)
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = x.view(x.size(0), -1)
		output = self.out(x)
		return output


print("Loading Doc2vec model")
model = gensim.models.Doc2Vec.load("models/doc2vec")
x_train = model.docvecs.vectors_docs
print("\nReading y_train\n")
y_train = []
with open("../dataset/audio_train.json", "r") as f:
	for line in tqdm(f):
		temp = json.loads(line)
		y_train.append(temp["overall"])

print("Loading Doc2vec of dev data")
x_dev, y_dev = pickle.load(open("dev_data.doc2vec", "rb"))

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

dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
cnn = CNN()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


for epoch in range(EPOCH):
	print("EPOCH: %d" %(epoch))
	for step, (x, y) in enumerate(train_loader):
		b_x = Variable(x)
		b_y = Variable(y)

		print("Output by CNN")
		output = cnn(b_x)[0] # output of the cnn
		print("Calculating loss")
		loss = loss_func(output, b_y) # loss
		print("Claering old gradients")
		optimizer.zero_grad() # clearing gradients
		print("Backpropogating")
		loss.backward() # backpropogation
		print("applygradients")
		optimizer.step() # applygradients
	
		if step % 80 == 0:
			test_output, last_layer = cnn(x_dev)
			pred_y = torch.max(test_output, 1)[1].data.squeeze()
			accuracy = sum(pred_y == y_dev) / float(y_dev.size(0))
			print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| dev accuracy: %.2f' % accuracy)

	torch.save(cnn, "models/cnn")