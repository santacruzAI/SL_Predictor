import pandas as pd
import numpy as np
import torch
import pickle
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import ray.tune as tune
import matplotlib.pyplot as plt

SEED = 77

def load_data():
  """Helper function to load data for model"""
  with open('split_data.p', 'rb') as file:
    x_train = pickle.load(file)
    y_train = pickle.load(file)
    x_test = pickle.load(file)
    y_test = pickle.load(file)
    class_weight = pickle.load(file)
  return (x_train, y_train, x_test, y_test, class_weight)

class Model(nn.Module):
  def __init__(self, x_train, y_train, x_test, y_test, class_weight):
    super(Model, self).__init__()
    # Network architecture
    self.input = nn.Linear(112, 30)  
    self.fc_1 = nn.Linear(30, 30)
    self.output = nn.Linear(30, 1)
    self.class_weight = torch.FloatTensor([class_weight])
    self.criterion = nn.BCELoss(weight=self.class_weight)
    
    # Training setup
    self.num_epochs = 0
    self.loss_history = []
    self.acc_history = []
    self.auc_history = []
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.train_results = np.zeros(self.y_train.shape)  # holds final label predictions
  
  def forward(self, x):
    x = self.input(x)
    x = nn.SiLU()(x)
    x = self.fc_1(x)
    x = nn.SiLU()(x)
    x = self.output(x)
    x= nn.Sigmoid()(x)
    return x 
  
  def Train(self, lr:float=0.0025, epochs:int=25, decay:float=0.0001, store_result:bool=False):
    """Fits the model to the training data."""
    self.train() 
    self.num_epochs = epochs
    num_entries = self.x_train.shape[0]
    optimizer = Adam(self.parameters(), lr=lr, weight_decay=decay)
    loss_over_time = []
    results = np.empty(self.y_train.shape)
    file = open('Train_results.txt', 'w')
    for epoch in range(self.num_epochs):
      loss_sum = 0
      acc = 0
      # determine the loss for each sample
      for i in range(num_entries):
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
      acc /= num_entries
      auc = roc_auc_score(self.y_train.cpu().detach().numpy(), results)
      
      # Store results of epoch
      self.loss_history.append(loss_sum/num_entries)
      self.acc_history.append(acc)
      self.auc_history.append(auc)
      if store_result == True:
        file.write('Epoch [%d/%d]\tLoss:%.4f\tAcc: %.4f\tAUROC: %.4f\n' % (epoch + 1, epochs, loss_sum, acc, auc))
      
    self.train_results = results
    file.close()
 
    
  def plot_history(self):
    hist_fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    hist_fig.suptitle('Training History')
    ax1.plot(np.arange(self.num_epochs), self.loss_history)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    ax2.plot(np.arange(self.num_epochs), self.acc_history)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
     
    ax3.plot(np.arange(self.num_epochs), self.auc_history)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AUROC')
      
    hist_fig.savefig(fname='train_history.png')      
      
  def evaluate(self):
    """
    Evaluates the model on the test set and returns the accuracy
    """
    self.eval()
    acc = 0
    count = 0
    results = np.empty(self.y_test.shape)
    loss = 0
    for (X, y) in zip(self.x_test, self.y_test):
      out = self.forward(X)
      results[count] = out.item()
      if (out.item()<0.5 and y.item() == 0) or (out.item()>=0.5 and y.item()==1):
          acc += 1
      loss = self.criterion(out, y).item()
      count += 1
    auc = roc_auc_score(self.y_test.cpu().detach().numpy(), results)
    return(loss, acc/len(self.x_test), auc)
  
  def predict_one(self, pair):
    """Given the function assciations of a gene pair, predicts whether 
       the pair will be SL or not.
       pair: encoded gene pair 
       Return: 1 if predicted to be SL or 0 otherwise
    """
    self.eval()
    pred = self.forward(pair)
    if (pred.item() >= 0.5):
      return 1
    else:
      return 0
 
def get_kf(k, x_train, y_train):
  lr = config["lr"]
  epochs = config["epochs"]
  decay = config["decay"]
  accuracy = []
  auroc = []
  kf = StratifiedKFold(n_splits=k, random_state=SEED)
  return kf

def KFolds(config, kf, tuning:bool=True):
  """Performs k-folds cross validation using the hyperparameters specified
     by config.
  """
  x_train = config[x_train]
  y_train = config[y_train]
  for train_idk, val_idx in kf.split(x_train, y_train):
    X_val, y_val = torch.FloatTensor(x_train[val_idx]), torch.FloatTensor(y_train[val_idx])
    X, y = torch.FloatTensor(np.delete(x_train, val_idx, axis=0)), torch.FloatTensor(np.delete(y_train, val_idx, axis=0))
    model = Model(X, y, X_val, y_val, config[class_weight])
    model.Train(lr, epochs, decay)
    auc, acc = model.evaluate(X_val, y_val)
    accuracy.append(acc)
    auroc.append(auc)
  if tuning == True:
    tune.report(mean_auc=sum(auroc)/k) 


def Tune(k, x_train, y_train, class_weight):
  """Selects best hyperparameters based on the mean accuracy values
  of K-folds cross validation for each combination of parameters
  as part of a grid search.
  """

  config={
  "x_train": x_train,
  "y_train": y_train,
  "class_weight": class_weight,
  "kf": get_kf(k, x_train, y_train),
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
    
def main():
  (x_train, y_train, x_test, y_test, class_weight) = load_data()
  #best_params = Tune(x_train, y_train, class_weight)
  best_params = {'lr':0.0025, 'epochs':25, 'decay':0.0075}
  print("Hyperparameters: ", best_params)

  # Train the final model using the tuned parameters
  model = Model(x_train, y_train, x_test, y_test, class_weight)
  model.Train(lr=best_params['lr'], epochs=best_params['epochs'], 
              decay=best_params['decay'], store_result = True)
            
  model.plot_history()

  print("Final train loss: ", model.loss_history[-1])
  print("Final train AUROC: ", model.auc_history[-1])
  print("Final train accuracy: ", model.acc_history[-1])

  test_loss, test_acc, test_auc = model.evaluate()
  print("Test loss: ", test_loss)
  print("Test AUROC: ", test_auc)
  print("Test accuracy: ", test_acc)

  # save the trained model
  torch.save(model.state_dict(), 'trained_model.pt')
    
if __name__ == "__main__":
  main()
