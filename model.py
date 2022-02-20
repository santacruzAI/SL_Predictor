import pandas as pd
import numpy as np
import torch
import pickle
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import ray.tune as tune
import matplotlib.pyplot as plt

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

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    # Network architecture
    self.input = nn.Linear(112, 30)  
    self.fc_1 = nn.Linear(30, 30)
    self.output = nn.Linear(30, 1)
    self.criterion = nn.BCELoss()    
  
  def forward(self, x):
    x = self.input(x)
    x = nn.ReLU()(x)
    x = self.fc_1(x)
    x = nn.ReLU()(x)
    x = self.output(x)
    x= nn.Sigmoid()(x)
    return x 
  
  def Train(self, x_train, y_train, lr = 0.01, epochs:int=100, store_result:bool=False, plot_loss:bool=False):
    """Fits the model to the training data."""
    self.train() 
    optimizer = Adam(self.parameters(), lr=lr, weight_decay=0.0001)
    loss_over_time = []
    for epoch in range(epochs):
      loss_sum = 0
      acc = 0
      for X, y in zip(x_train, y_train):
        optimizer.zero_grad()
        out = self.forward(X)
        if (out.item()<0.5 and y.item() == 0) or (out.item()>=0.5 and y.item()==1):
          acc += 1
        loss = self.criterion(out, y)
        if plot_loss == True:
          loss_sum += loss.item()
        loss.backward()
        optimizer.step()
      acc /= len(x_train)
      if (store_result == True):
        with open('Train_results.txt', 'w') as file:
          file.write('Epoch [%d/%d]\tLoss:%.4f\tAcc: %.4f\n' % (epoch + 1, epochs, loss.item(), acc))
      if plot_loss == True:
          loss_over_time.append(loss_sum)
    if plot_loss == True:
      plt.plot(np.linspace(1, epochs, num=epochs), np.array(loss_over_time))
      plt.title("Loss for lr={}".format(lr))
      plt.xlabel("Epochs")
      plt.ylabel("Loss")
      plt.show()
      
      
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
 

def KFolds(config, tune:bool=True):
  """Performs k-folds cross validation using the parameters specified
  by config
  """
  k = 6
  lr = config["lr"]
  epochs = config["epochs"]
  accuracy = [] # correspondig accuracies of the models
  kf = StratifiedKFold(n_splits=k)
  for train_idk, val_idx in kf.split(x_train, y_train):
    model = Model()
    X_val, y_val = torch.FloatTensor(x_train[val_idx]), torch.FloatTensor(y_train[val_idx])
    X, y = torch.FloatTensor(np.delete(x_train, val_idx, axis=0)), torch.FloatTensor(np.delete(y_train, val_idx, axis=0))
    model.Train(X, y, lr, epochs, plot_loss=False)
    accuracy.append(model.evaluate(X_val, y_val))
  if tune == True
  tune.report(mean_accuracy=sum(accuracy)/k) 


def tune():
  """Selects best hyperparameters based on the mean accuracy values
  of K-folds cross validation for each combination of parameters
  as part of a grid search.
  """
  config={
  "lr": tune.grid_search([0.008,0.005, 0.001, 0.012]),
  "epochs": tune.grid_search([50, 100, 150, 200, 300])
  }
  analysis = tune.run(
    KFolds,
    config = config,
    metric="mean_accuracy",  
    mode="max",
    num_samples=1)
  best_params = analysis.best_config()
  return best_params
    
best_params = tune()
print("Tuned hyperparameters: ", best_params)

# Train the final model using the tuned parameters
model = Model()
model.train(x_train, y_train, lr=best_params["lr"], epochs=best_params["epochs"], store_result = True)