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
from model import Model

# Load trained model
file = open("trained_model.p", "rb")
model = pickle.load(file)
file.close()

# t-SNE ##################################################
results = np.squeeze(model.results, 1)
# Get predicted labels
class_pred = np.where(results >= 0.5, np.ones(results.shape), np.zeros(results.shape)) # label predictions based on raw probabilities

# Perform t-SNE dimensionality reduction
m = TSNE(init='pca', method='exact', n_iter=3000, learning_rate=100, perplexity=30, random_state=77)
tsne_features = m.fit_transform(model.x_train)
tsne_df = pd.DataFrame({'x':tsne_features[:, 0], 'y':tsne_features[:, 1], "SL Prediction":class_pred})

# Plot t-SNE results
sns.scatterplot(x='x', y='y', hue='SL Prediction', palette=['red', 'blue'], legend='full', data=tsne_df)
plt.show()