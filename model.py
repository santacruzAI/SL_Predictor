import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
import pickle
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

file = open("data.p", "rb")
data_table = pickle.load(file)
df = pd.read_pickle("data.p")
file.close()

X = df.loc[:, df.columns.values[0:-1]].to_numpy()
y = df.loc[:, 'SL'].to_numpy()

# Split data into training and test sets 
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=0)
# Convert to tensors
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test)

epochs = 100
class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    # Network architecture
    self.input = nn.Linear(112, 150)  
    self.fc_1 = nn.Linear(150, 100)
    self.output = nn.Linear(100, 1)
    self.criterion = nn.BCELoss()    
  
  def forward(self, x):
    x = self.input(x)
    x = nn.ReLU()(x)
    x = self.fc_1(x)
    x = nn.ReLU()(x)
    x = self.output(x)
    x= nn.Sigmoid()(x)
    return x 
    
  def fit(self, x_train, y_train, p:bool=False):
    """ 
    Fits the model to the training data.
    """
    self.train() 
    optimizer = Adam(model.parameters())
    # Training loop
    accuracy = []
    for epoch in range(epochs):
      acc = 0
      for X, y in zip(x_train, y_train):
        optimizer.zero_grad()
        out = self.forward(X)
        if (out.item()<0.5 and y.item() == 0) or (out.item()>=0.5 and y.item()==1):
          acc += 1
        loss = self.criterion(out, y)
        loss.backward()
        optimizer.step()
      acc /= len(x_train)
      accuracy.append(acc)
      if (p == True):
        print('Epoch [%d/%d]\tLoss:%.4f\tAcc: %.4f' % (epoch + 1, epochs, loss.item(), acc))
  
  def evaluate(self, x_test, y_test):
    """
    Evaluates the model on the test set and returns the accuracy
    """
    self.eval()
    acc = 0
    for (X, y) in zip(x_test, y_test):
      out = self.forward(X)
      if (out.item()<0.5 and y.item() == 0) or (out.item()>=0.5 and y.item()==1):
          acc += 1
      loss = self.criterion(out, y)
    return(acc/len(x_test))
 

# model initialization

# K-folds cross validation
k = 10
accuracy = []
kf = StratifiedKFold(n_splits=k)

for train_idk, val_idx in kf.split(x_train, y_train):
  model = Model()
  X_val, y_val = torch.FloatTensor(x_train[val_idx]), torch.FloatTensor(y_train[val_idx])
  X, y = torch.FloatTensor(np.delete(x_train, val_idx, axis=0)), torch.FloatTensor(np.delete(y_train, val_idx, axis=0))
  model.fit(X, y)
  accuracy.append(model.evaluate(X_val, y_val))
print(accuracy)
print("Average model accuracy: ", sum(accuracy)/len(accuracy))