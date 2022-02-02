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
    self.criterion = nn.BCELoss()
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
    
  def fit(self, train_data):
    """ 
    Fits the model to the training data.
    """
    self.train() 
    optimizer = Adam(model.parameters())
    # Training loop
    accuracy = []
    for epoch in range(epochs):
      acc = 0
      for (X, y) in train:
        optimizer.zero_grad()
        out = self.forward(X)
        if (out.item()<0.5 and y.item() == 0) or (out.item()>=0.5 and y.item()==1):
          acc += 1
        loss = self.criterion(out, y)
        loss.backward()
        optimizer.step()
      acc /= len(train)
      accuracy.append(acc)
      print('Epoch [%d/%d]\tLoss:%.4f\tAcc: %.4f' % (epoch + 1, epochs, loss.item(), acc))
  
  def evaluate(self, test_data):
    self.eval()
    acc = 0
    for (X, y) in test_data:
      out = self.forward(X)
      if (out.item()<0.5 and y.item() == 0) or (out.item()>=0.5 and y.item()==1):
          acc += 1
      loss = self.criterion(out, y)
    return(acc/len(test_data))
      
    


# model initialization
model = Model()
model.fit(train)
  
# Grid search for hyperparameters
# K-fold