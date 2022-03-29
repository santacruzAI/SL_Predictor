"""
Creates visuals for the trained model.
Visuals include:
  1. t-SNE depiction of function relations
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from model import *

SEED = 77

def tSNE(model):
  """Creates a t-SNE plot for a given trained model"""
  results = np.squeeze(model.train_results, 1)
  # Get predicted labels
  class_pred = np.where(results >= 0.5, np.ones(results.shape), np.zeros(results.shape)) # label predictions based on raw probabilities

  # Perform t-SNE dimensionality reduction
  m = TSNE(init='pca', method='exact', n_iter=1000, learning_rate=200, perplexity=15, random_state=SEED)
  tsne_features = m.fit_transform(model.x_train)
  tsne_df = pd.DataFrame({'x':tsne_features[:, 0], 'y':tsne_features[:, 1], "SL Prediction":class_pred})
  # Plot t-SNE results
  sns.scatterplot(x='x', y='y', hue='SL Prediction', palette=['red', 'blue'], legend='full', data=tsne_df)
  plt.savefig("tsne.png")

def main():
  # load data
  (x_train, y_train, x_test, y_test, class_weight) = load_data()
  # Load trained model
  model = Model(x_train, y_train, x_test, y_test, class_weight)
  model.load_state_dict(torch.load('trained_model.pt'))
  model.Train()
  tSNE(model)

if __name__ == "__main__":
  main()
