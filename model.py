import pandas as pd
import numpy as np
import torch
import pickle
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import ray.tune as tune
from datetime import datetime

start_time = datetime.now()

file = open("data.p", "rb")
data_table = pickle.load(file)
df = pd.read_pickle("data.p")
file.close()

X = df.loc[:, df.columns.values[0:-1]].to_numpy()
y = df.loc[:, 'SL'].to_numpy()

# Split data into training and test sets 
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=0)
num_inputs = x_train
# Convert to tensors
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test)

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    # Network architecture
    self.input = nn.Linear(num_inputs, num_inputs*2)  
    self.fc_1 = nn.Linear(num_inputs*2, num_inputs/2)
    self.output = nn.Linear(num_inputs/2, 1)
    self.criterion = nn.BCELoss()    
  
  def forward(self, x):
    x = self.input(x)
    x = nn.ReLU()(x)
    x = self.fc_1(x)
    x = nn.ReLU()(x)
    x = self.output(x)
    x= nn.Sigmoid()(x)
    return x 
  
  def train(self, x_train, y_train, lr = 0.01, epochs:int=100, store_result:bool=False):
    """ 
    Fits the model to the training data.
    """
    print("Start training at {}".format(datetime.now()))
    self.train() 
    optimizer = Adam(self.parameters(), lr=lr)
    # Training loop
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
      if (store_result == True):
        with open('Train_results.txt', 'w') as file:
          file.write('Epoch [%d/%d]\tLoss:%.4f\tAcc: %.4f\n' % (epoch + 1, epochs, loss.item(), acc))
    print("End training at {}".format(datetime.now()))
  
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
count = 0
# K-folds cross validation
def KFolds(config):
  print("Starting KFolds")
  k = 6
  lr = config["lr"]
  epochs = config["epochs"]
  accuracy = [] # correspondig accuracies of the models
  kf = StratifiedKFold(n_splits=k)
  for train_idk, val_idx in kf.split(x_train, y_train):
    model = Model()
    X_val, y_val = torch.FloatTensor(x_train[val_idx]), torch.FloatTensor(y_train[val_idx])
    X, y = torch.FloatTensor(np.delete(x_train, val_idx, axis=0)), torch.FloatTensor(np.delete(y_train, val_idx, axis=0))
    model.train(X, y, lr, epochs)
    accuracy.append(model.evaluate(X_val, y_val))
  count += 1
  tune.report(mean_accuracy=sum(accuracy)/k)
  print("Finished ", count, " search(s) by ", datetime.now())
  
# tune on Kfolds function and use average accuracy as metric to improve. result is list of best parameters.
# use best parameters to train a new model on all of the training data. Lastly, evaluate this model on test set.
# Parameters to tune:
# model depth
# model width
# learning rate
# number of epochs
# number of folds for cross validation

config={
  "lr": tune.grid_search([0.001,0.01, 0.1]),
  "epochs": tune.grid_search([50, 100, 150, 200])#,
  #"depth": tune.grid_search([1, 2, 4, 8, 16])
}

analysis = tune.run(
  KFolds,
  config = config,
  metric="mean_accuracy",  
  mode="max",
  num_samples=1)
    
print("here")
best_params = analysis.best_config()
print("Tuned hyperparameters: ", best_params)
# Train a new model using the tuned parameters
model = Model()
model.train(x_train, y_train, lr=best_params["lr"], epochs=best_params["epochs"], store_result = True)
finish_time = datetime.now()
print("Total time: ", finish_time - start_time)