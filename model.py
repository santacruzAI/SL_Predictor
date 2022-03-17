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
import matplotlib.pyplot as plt

with open('data.p', 'rb') as file:
  data_table = pickle.load(file)
  df = pd.read_pickle("data.p")

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
# account for class imbalance
class_weights = torch.FloatTensor([0.88])

class Model(nn.Module):
  def __init__(self, x_train, y_train):
    super(Model, self).__init__()
    # Network architecture
    self.input = nn.Linear(112, 30)  
    self.fc_1 = nn.Linear(30, 30)
    self.output = nn.Linear(30, 1)
    self.criterion = nn.BCELoss(weight=class_weights)
    
    # Training set up
    self.x_train = x_train
    self.y_train = y_train
    self.num_entries = self.x_train.shape[0]
    self.num_features = self.x_train.shape[1]
    self.num_epochs = 0
    self.loss_history = []
    self.acc_history = []
    self.auc_history = []
    self.train_results = np.zeros(y_train.shape)  # holds final label predictions
  
  def forward(self, x):
    x = self.input(x)
    x = nn.SiLU()(x)
    x = self.fc_1(x)
    x = nn.SiLU()(x)
    x = self.output(x)
    x= nn.Sigmoid()(x)
    return x 
  
  def Train(self, lr = 0.01, epochs:int=100, decay=0.0001, store_result=False):
    """Fits the model to the training data."""
    self.train() 
    self.num_epochs = epochs
    optimizer = Adam(self.parameters(), lr=lr, weight_decay=decay)
    loss_over_time = []
    results = np.empty(self.y_train.shape)
    for epoch in range(self.num_epochs):
      loss_sum = 0
      acc = 0
      # determine the loss for each sample
      for i in range(self.num_entries):
        optimizer.zero_grad()
        out = self.forward(self.x_train[i,:])
        results[i] = out.item()  # Store raw probability
        # determine the prediction accuracy
        if (out.item()<0.5 and self.y_train[i].item() == 0) or (out.item()>=0.5 and self.y_train[i].item()==1):
          acc += 1
        loss = self.criterion(out, self.y_train[i])  # calculate loss
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
      # Determine final accuracy and AUROC scores for epoch
      acc /= self.num_entries
      auc = roc_auc_score(self.y_train.cpu().detach().numpy(), results)
      
      # Store results of epoch
      self.loss_history.append(loss_sum/self.num_entries)
      self.acc_history.append(acc)
      self.auc_history.append(auc)
      if store_result == True:
        with open('Train_results.txt', 'w') as file:
          file.write('Epoch [%d/%d]\tLoss:%.4f\tAcc: %.4f\tAUROC: %.4f\n' % (epoch + 1, epochs, loss_sum, acc, auc))
    self.train_results = results
 
    
    def plot_history(self):
      hist_fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
      hist_fig.supitle('Training History')
      ax1.plot(np.arange(self.num_epochs), self.loss_history)
      ax1.set_title('Train loss')
      ax1.set_xlabel('Epoch')
      ax1.set_ylabel('Loss')
      
      ax2.plot(np.arange(self.num_epochs), self.acc_history)
      ax2.set_title('Train Accuracy')
      ax2.set_xlabel('Epoch')
      ax2.set_ylabel('Loss')
      
      ax3.plot(np.arange(self.num_epochs), self.auc_history)
      ax3.set_title('Train AUROC')
      ax3.set_xlabel('Epoch')
      ax3.set_ylabel('Loss')
      
      hist_fig.savefig(fname='train_history.png')      
      
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
  decay = config["decay"]
  accuracy = []
  auroc = []
  kf = StratifiedKFold(n_splits=k)
  for train_idk, val_idx in kf.split(x_train, y_train):
    X_val, y_val = torch.FloatTensor(x_train[val_idx]), torch.FloatTensor(y_train[val_idx])
    X, y = torch.FloatTensor(np.delete(x_train, val_idx, axis=0)), torch.FloatTensor(np.delete(y_train, val_idx, axis=0))
    model = Model(x_train=X, y_train=y)
    model.Train(lr, epochs, decay)
    auc, acc = model.evaluate(X_val, y_val)
    accuracy.append(acc)
    auroc.append(auc)
  if tuning == True:
    tune.report(mean_auc=sum(auroc)/k) 


def Tune():
  """Selects best hyperparameters based on the mean accuracy values
  of K-folds cross validation for each combination of parameters
  as part of a grid search.
  """
  config={
  "lr": tune.grid_search([0.0005, 0.00075, 0.001, 0.0025, 0.005]),
  "epochs": tune.grid_search([25, 50, 75, 100, 125]),
  "decay": tune.grid_search([0.0001, 0.0005, 0.001, 0.005, 0.01])
  }
  analysis = tune.run(
    KFolds,
    config = config,
    metric="mean_auc",  
    mode="max",
    num_samples=1,
    resources_per_trial={"gpu":1})
  best_trial = analysis.get_best_trial("mean_auc", "max", "last")
  return best_trial.config
    
best_params = Tune()
print("Best hyperparameters: ", best_params)

# Train the final model using the tuned parameters
model = Model(x_train, y_train)
model.Train(x_train, y_train, lr=best_params['lr'], epochs=best_params['epochs'], 
            decay=best_params['decay'] store_result = True)
            
model.plot_history()

print("Final train loss: ", model.train_history[-1])
print("Final train AUROC: ", model.auc_history[-1])
print("Final train accuracy: ", model.acc_history[-1])

test_auc, test_acc = model.evaluate(x_test, y_test)
print("Test AUROC: ", test_auc)
print("Test accuracy: ", test_acc)

with open("trained_model.p", "wb") as file:
  pickle.dump(model, file)
