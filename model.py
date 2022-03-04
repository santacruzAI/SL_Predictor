import pandas as pd
import numpy as np
import torch
import pickle
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import ray.tune as tune

file = open("data.p", "rb")
data_table = pickle.load(file)
df = pd.read_pickle("data.p")
file.close()

X = df.loc[:, df.columns.values[0:-1]].to_numpy()
y = df.loc[:, 'SL'].to_numpy()

# Split data into training and test sets 
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=77)

# Prepare data for later visualization
feature_data = x_train[:, 2:]

# Convert to tensors
train_genes = x_train[:,0:2]
test_genes = x_test[:,0:2]
x_train = torch.FloatTensor(x_train[:,2:].astype('float64'))
y_train = torch.FloatTensor(y_train.astype('float64'))
x_test = torch.FloatTensor(x_test[:,2:].astype('float64'))
y_test = torch.FloatTensor(y_test.astype('float64'))
class_weights = torch.FloatTensor([0.88])

class Model(nn.Module):
  def __init__(self, x_train, y_train):
    super(Model, self).__init__()
    # Network architecture
    self.input = nn.Linear(112, 30)  
    self.fc_1 = nn.Linear(30, 30)
    self.output = nn.Linear(30, 1)
    self.criterion = nn.BCELoss(weight=class_weights)
    self.train_results = np.zeros(y_train.shape)  # holds final labele predictions
    self.test_results = np.zeros(y_test.shape) 
    self.x_train = x_train
    self.y_train = y_train
  
  def forward(self, x):
    x = self.input(x)
    x = nn.SiLU()(x)
    x = self.fc_1(x)
    x = nn.SiLU()(x)
    x = self.output(x)
    x= nn.Sigmoid()(x)
    return x 
  
  def Train(self, x_train, y_train, lr = 0.01, epochs:int=100, store_result:bool=False, plot_loss:bool=False):
    """Fits the model to the training data."""
    num_entries = x_train.shape[0]
    num_features = x_train.shape[1]
    self.train() 
    optimizer = Adam(self.parameters(), lr=lr, weight_decay=0.0001)
    loss_over_time = []
    results = np.empty(y_train.shape)
    for epoch in range(epochs):
      loss_sum = 0
      acc = 0
      for i in range(num_entries):
        optimizer.zero_grad()
        out = self.forward(x_train[i,:])   
        results[i] = out.item()
        if (out.item()<0.5 and y_train[i].item() == 0) or (out.item()>=0.5 and y_train[i].item()==1):
          acc += 1
        loss = self.criterion(out, y_train[i])
        if plot_loss == True:
          loss_sum += loss.item()
        loss.backward()
        optimizer.step()
      acc /= len(x_train)
      auc = roc_auc_score(y_train.cpu().detach().numpy(), results)
      if (store_result == True):
        with open('Train_results.txt', 'w') as file:
          file.write('Epoch [%d/%d]\tLoss:%.4f\tAcc: %.4f\n' % (epoch + 1, epochs, loss.item(), acc))
      if plot_loss == True:
          loss_over_time.append(loss_sum)
      if epoch % 10 == 0:
        print("epoch: ", epoch, " auc: ", auc)
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
    count = 0
    results = np.empty(y_test.shape)
    for (X, y) in zip(x_test, y_test):
      out = self.forward(X)
      results[count] = out.item()
      if (out.item()<0.5 and y.item() == 0) or (out.item()>=0.5 and y.item()==1):
          acc += 1
      loss = self.criterion(out, y)
      count += 1
    auc = roc_auc_score(y_test.cpu().detach().numpy(), results)
    return(auc, acc/len(x_test))
  
  def predict_one(self, pair):
    """Given a gene pair, predicts whether it will be SL or not"""
    pass
    
  def predict_many(self, data):
    """Params: data = list of gene pairs
    """
    pass
 

def KFolds(config, tuning:bool=True):
  """Performs k-folds cross validation using the parameters specified
  by config
  """
  k = 6
  lr = config["lr"]
  epochs = config["epochs"]
  accuracy = []
  auc = []
  kf = StratifiedKFold(n_splits=k)
  for train_idk, val_idx in kf.split(x_train, y_train):
    model = Model()
    X_val, y_val = torch.FloatTensor(x_train[val_idx]), torch.FloatTensor(y_train[val_idx])
    X, y = torch.FloatTensor(np.delete(x_train, val_idx, axis=0)), torch.FloatTensor(np.delete(y_train, val_idx, axis=0))
    model.Train(X, y, lr, epochs, plot_loss=False)
    auc, acc = model.evaluate(X_val, y_val)
    accuracy.append(acc)
    auc.append(auc)
  if tuning == True:
    tune.report(mean_auc=sum(auc)/k) 


def Tune():
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
    metric="mean_auc",  
    mode="max",
    num_samples=1)
  return analysis.best_config()
    
#best_params = Tune()
#print("Tuned hyperparameters: ", best_params)

# Train the final model using the tuned parameters
model = Model(x_train, y_train)
#model.Train(x_train, y_train, lr=best_params["lr"], epochs=best_params["epochs"], store_result = True)
model.Train(x_train, y_train, lr=0.012, epochs=100)
train_auc, train_acc = model.evaluate(x_train, y_train)
print("Train AUROC: ", train_auc)
print("Train accuracy: ", train_acc)
test_auc, test_acc = model.evaluate(x_test, y_test)
print("Test AUROC: ", test_auc)
print("Test accuracy: ", test_acc)

file = open("trained_model.p", "wb")
pickle.dump(model, file)
file.close()
