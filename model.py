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
df = pd.read_pickle("data.p")
file.close()
print(df)
#df = pd.DataFrame(data_table, columns = data_table[0])
X = df.loc[:, df.columns.values[0:-1]].to_numpy()
y = df.loc[:, 'SL'].to_numpy()

print(X)
class SL_Pairs(Dataset):
  def __init__(self, X, y):
    # convert to tensors
    self.functs = torch.FloatTensor(X)
    self.labels = torch.FloatTensor(y)

    # Why would we need another dimension? for bias? only need bias for features
    #self.functs = self.functs.unsqueeze(1) # add extra dimension for torch
    #self.labels = self.labels.unsqueeze(1)

  def __len__(self):
    assert len(self.functs) == len(self.labels) # ensure 1-to-1 correspondence
    return len(self.labels)

  def __getitem__(self, i):
    return self.functs[i], self.labels[i] # return X, y pair


    
data = SL_Pairs(X, y)
trainCount = int(0.8 * len(data)) # percent of data for training
train, test = random_split(data, [trainCount, len(data) - trainCount])

epochs = 100
class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    # Network architecture
    self.input = nn.Linear(112, 150)  
    self.fc_1 = nn.Linear(150, 100)
    self.output = nn.Linear(100, 1)
    # self.input = nn.Linear(112, 150)  
    # self.fc_1 = nn.Linear(150, 300)
    # self.fc_2 = nn.Linear(300, 450)
    # self.fc_3 = nn.Linear(450, 300)
    # self.fc_4 = nn.Linear(300, 150)
    # self.output = nn.Linear(150, 1)
    
  
  def forward(self, x):
    x = self.input(x)
    x = nn.ReLU()(x)
    x = self.fc_1(x)
    x = nn.ReLU()(x)
    # x = self.fc_2(x)
    # x = nn.ReLU()(x)
    # x = self.fc_3(x)
    # x = nn.ReLU()(x)
    # x = self.fc_4(x)
    # x = nn.ReLU()(x)
    x = self.output(x)
    x= nn.Sigmoid()(x)
    return x 

# model initialization
model = Model()
model.train()
criterion = nn.BCELoss() #nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())
#for x in model.parameters():
  #print(x)

# Training loop
accuracy = []
assert 0 == 0.0
for epoch in range(0, epochs):
  acc = 0
  for (X, y) in train:
    optimizer.zero_grad()
    out = model(X)
    if (out.item()<0.5 and y.item() == 0) or (out.item()>=0.5 and y.item()==1):
      acc += 1
    #print("predicted: ", out.item(), " actual: ", y.item())
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
  acc /= len(train)
  accuracy.append(acc)
  print('Epoch [%d/%d]\tLoss:%.4f\tAcc: %.4f' % (epoch + 1, epochs, loss.item(), acc))
  
  