import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
import pickle
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, random_split

file = open("data.p", "rb")
data_table = pickle.load(file)
file.close()

df = pd.DataFrame(data_table, columns = ['GeneA Function', 'GeneB Function', 'SL'])
X = df.loc[:, 'GeneA Function':'GeneB Function'].to_numpy()
y = df.loc[:, 'SL'].to_numpy()

class SL_Pairs(Dataset):
  def __init__(self, X, y):
    # convert to tensors
    self.functs = torch.from_numpy(X)
    self.labels = torch.from_numpy(y)

    self.functs = self.functs.unsqueeze(1) # add extra dimension for torch
    self.labels = self.labels.unsqueeze(1)
    print(self.functs.shape)

  def __len__(self):
    assert len(self.functs) == len(self.labels) # ensure 1-to-1 correspondence
    return len(self.labels)

  def __getitem__(self, i):
    return self.functs[i], self.labels[i] # return X, y pair


    
data = SL_Pairs(X, y)
trainCount = int(0.8 * len(data)) # percent of data for training
train, test = random_split(data, [trainCount, len(data) - trainCount])

print(train)
#print(train.shape)
train[:, 0] = torch.FloatTensor(train[:, 0])
train[:, 1] = torch.LongTensor(train[:, 1])
test[:, 0] = torch.FloatTensor(test[:, 0])
test[:, 1] = torch.LongTensor(test[:, 1])

# input looks like [function1, function2]


epochs = 100
class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    # Network architecture
    self.input = nn.Linear(1, 2, 150)  #150 = dimension of output
    self.fc_1 = nn.Linear(150, 100)
    self.output = nn.Linear(100, 2)
  
  def forward(self, x):
    x = self.input(x)
    x = nn.ReLu()(x)
    x = self.fc_1(x)
    x = nn.ReLu()(x)
    x = self.output(x)
    x = nn.Sigmoid(x)
    return x 

# model initialization
model = Model()
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

# Training loop
accuracy = []
for epoch in range(0, epochs):
  acc = 0
  for (X, y) in train:
    print(X, y)
    optimizer.zero_grad()
    out = model(X)
    if out.argmax().item() == label.item():
      acc += 1
    loss = criterion(out, label)
    loss.backward()
    optimizer.step()
  acc /= len(train)
  accuracy.append(acc)
  print('Epoch [%d/%d]\tLoss:%.4f\tAcc: %.4f' % (epoch + 1, EPOCHS, loss.item(), acc))
  
  