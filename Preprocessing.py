import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
import pickle
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
[all_genes.append(i) for i in df_funct['GeneSym'] if i not in all_genes]
print(len(all_genes))

functions = []
[functions.append(i[1:]) for i in df_funct['Child1 Function'] if i[1:] not in functions]
print(len(functions))  

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

funct_dict = {}
for g, f in zip(df_funct['GeneSym'], df_funct['Child1 Function']):
  if g in pos_genes or g in neg_genes:
    #Add to function dictionary
    if g not in funct_dict.keys():
      funct_dict[g] = [f[1:]]
    else:
      funct_dict[g] = funct_dict[g] + [f[1:]]

data = []
for (x,y) in pos_pairs:
  l1 = [1 if f in funct_dict[x] else 0 for f in functions]
  l2 = [1 if f in funct_dict[y] else 0 for f in functions]
  data.append([x]+[y]+l1+l2+[1])
for (x,y) in neg_pairs:
  l1 = [1 if f in funct_dict[x] else 0 for f in functions]
  l2 = [1 if f in funct_dict[y] else 0 for f in functions]
  data.append([x]+[y]+l1+l2+[0])

columns = [["Gene1"] + ["Gene2"] + ["Gene1 " + f for f in functions] + ["Gene2 " + f for f in functions] + ["SL"]]
dataset = pd.DataFrame(data=data, columns=columns)

print(dataset.shape)
print(dataset)
file = open("data.p", "wb")
pickle.dump(dataset, file)
file.close()   
  