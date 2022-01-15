import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, random_split

df_pos = pd.read_csv('Human_SL.csv')
df_neg = pd.read_csv('Human_nonSL.csv')
df_funct = pd.read_csv('FunctionMapping.txt', sep='~')

# Label columns
df_pos.columns=['geneA symbol', 'geneA ID', 'geneB symbol', 'geneB ID', 'Cell line', 'PubmedID', 'Source', 'Score']
df_neg.columns=['geneA symbol', 'geneA ID', 'geneB symbol', 'geneB ID', 'Cell line', 'PubmedID', 'Source', 'Score']
df_funct.columns = ['GeneSym', 'Parent Function ID', 'Parent Function', 'Child1 Function ID', 'Child1 Function', 'Child2 Function ID', 'Child2 Function']

all_genes = []  # List of genes from function dataset
all_genes = [i for i in df_funct['GeneSym'] if i not in all_genes]

# Condense data
# isolate data for K562 cell line and biologically based source
# only using gene pairs that are also in the function dataset
pos_genes = []  # genes from Human_SL.csv that are also in KO dataset
neg_genes = []

pos_pairs = []  # List of gene pair tuples
neg_pairs = []

# desired sources and cell lines from dataa
CL_desired = ['K562;K562', 'K562', 'K562;Jurkat', 'K562;K562;K562', 'K562;K562;K562;K562']
S_desired = ['GenomeRNAi', 'CRISPR/CRISPRi']

#where CL is cell line and S is source and A & B are genes
for A, B, CL, S in zip(df_pos['geneA symbol'], df_pos['geneB symbol'], df_pos['Cell line'], df_pos['Source']):
  if (A in all_genes) and (B in all_genes) and (CL in CL_desired) and (S in S_desired):
    # add genes to lists if not in them already
    if A not in pos_genes: 
      pos_genes.append(A)
    if B not in pos_genes:
      pos_genes.append(B)
    if (A, B) not in pos_pairs and (B, A) not in pos_pairs:
      pos_pairs.append((A, B))

print('number of positive genes: ', (len(pos_genes)))
print('number of positive gene pairs: ', (len(pos_pairs)))

# repeat for negative data
for A, B, CL, S in zip(df_neg['geneA symbol'], df_neg['geneB symbol'], df_neg['Cell line'], df_neg['Source']):
  if (A in all_genes) and (B in all_genes) and (CL in CL_desired) and (S in S_desired):
    # add genes to lists if not in them already
    if A not in neg_genes: 
      neg_genes.append(A)
    if B not in neg_genes:
      neg_genes.append(B)
    if (A, B) not in neg_pairs and (B, A) not in neg_pairs:
      neg_pairs.append((A, B))

print('number of negative genes: ', (len(neg_genes)))
print('number of negative gene pairs: ', (len(neg_pairs)))

prim_funct_dict = {}
sec_funct_dict = {}
tert_funct_dict = {}
for g, p, s, t in zip(df_funct['GeneSym'], df_funct['Parent Function ID'], df_funct['Child1 Function ID'], df_funct['Child2 Function ID']):
  if g in pos_genes or g in neg_genes:
    #Add to primary function dictionary
    if g not in prim_funct_dict.keys():
      prim_funct_dict[g] = [p]
    else:
      prim_funct_dict[g] = prim_funct_dict[g] + [p]
    #Add to secondary function dictionary
    if g not in sec_funct_dict.keys():
      sec_funct_dict[g] = [s]
    else:
      sec_funct_dict[g] = sec_funct_dict[g] + [s]
    #Add to tertiary function dictionary
    if g not in tert_funct_dict.keys():
      tert_funct_dict[g] = [t]
    else:
      tert_funct_dict[g] = tert_funct_dict[g] + [t]


"""
format Gene A parent function, Gene A child 1 function, Gene A child 2 function, Gene B parent function, Gene B child 1 function, Gene B child 2 function
Combine all parent, child1, and child2 function combinations into one list for each gene of every gene pair.
"""

pos_data = []
neg_data = []
combinations = []

# TOTAL COMBINATIONS ****************************************
# for (g1,g2) in pos_pairs[0:300]:
  # geneA = [[x,y,z] for x in prim_funct_dict[g1] for y in sec_funct_dict[g1] for z in tert_funct_dict[g1]]
  # geneB = [[x,y,z] for x in prim_funct_dict[g2] for y in sec_funct_dict[g2] for z in tert_funct_dict[g2]]
  # pos_data = pos_data + [i+ii+[1] for i in geneA for ii in geneB]  # concatenate the information for both genes in the pair
                                                        # # add the SL value (1 for positive genes)
# print(len(pos_data))

# #Repeat for negative pairs
# for (g1,g2) in neg_pairs[0:20]:
  # geneA = [[x,y,z] for x in prim_funct_dict[g1] for y in sec_funct_dict[g1] for z in tert_funct_dict[g1]]
  # geneB = [[x,y,z] for x in prim_funct_dict[g2] for y in sec_funct_dict[g2] for z in tert_funct_dict[g2]]
  
  # neg_data = neg_data + [i+ii+[0] for i in geneA for ii in geneB]  

# print(len(neg_data))
# combinations = pos_data + neg_data

## convert data to pandas dataframe
# df = pd.DataFrame(combinations, columns = ['GeneA Parent', 'GeneA Child1', 'GeneA Child2', 'GeneB Parent', 'GeneB Child1', 'GeneB Child2', 'SL'])
# df.head(n=5)
# X = df.loc[:, 'GeneA Parent':'GeneB Child2'].to_numpy()
# y = df.loc[:, 'SL'].to_numpy()


# ONLY USING MID LEVEL FUNCTION COMBINATIONS *****************************
for (g1, g2) in pos_pairs:
  pos_data = pos_data + [[x, y, 1] for x in sec_funct_dict[g1] for y in sec_funct_dict[g2]]

for (g1, g2) in neg_pairs:
  neg_data = neg_data + [[x, y, 0] for x in sec_funct_dict[g1] for y in sec_funct_dict[g2]]

combinations = pos_data + neg_data
print(len(combinations))
num_rows = 0.8*len(combinations)

df = pd.DataFrame(combinations, columns = ['GeneA Function', 'GeneB Function', 'SL'])
X = df.loc[:, 'GeneA Function':'GeneB Function'].to_numpy()
y = df.loc[:, 'SL'].to_numpy()

class SL_Pairs(Dataset):
  def __init__(self, X, y):
    # convert to tensors
    self.functs = torch.from_numpy(X)
    self.labels = torch.from_numpy(y)

    self.functs = self.functs.unsqueeze(1) # add extra dimension for torch
    self.labels = self.labels.unsqueeze(1)
    print(self.functs.shape)

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
    self.input = nn.Linear(2, 1, 150)  #150 = dimension of output
    self.fc_1 = nn.Linear(150, 100)
    self.output = nn.Linear(100, 2)
  
  def forward(self, x):
    x = self.input(x)
    x = nn.ReLu()(x)
    x = self.fc_1(x)
    x = nn.ReLu()(x)
    x = self.output(x)
    x = nn.Sigmoid(x)
    return x 

# model initialization
model = Model()
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

# Training loop
accuracy = []
for epoch in range(0, epochs):
  acc = 0
  for (X, y) in train:
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    print(X, y)
    optimizer.zero_grad()
    out = model(X)
    if out.argmax().item() == label.item():
      acc += 1
    loss = criterion(out, label)
    loss.backward()
    optimizer.step()
  acc /= len(train)
  accuracy.append(acc)
  print('Epoch [%d/%d]\tLoss:%.4f\tAcc: %.4f' % (epoch + 1, EPOCHS, loss.item(), acc))
    
  
  
  