import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
import pickle
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, random_split
from sklearn.model_selection import train_test_split

CLASS_WEIGHT = 0
SEED = 77

def parse_data(pos_data_file:str='Human_SL.csv', neg_data_file:str='Human_nonSL.csv', map_file:str='FunctionMapping.txt'):
  """
  Reads files and converts data to Pandas dataframes. 
  params: pos_data_file: name of file containing positive data entries.
          pos_data_file: name of file containing positive data entries.
  returns: three pandas dataframes for the positive data, negative data, and function data respectively.
  """
  df_pos = pd.read_csv('Human_SL.csv')
  df_neg = pd.read_csv('Human_nonSL.csv')
  df_funct = pd.read_csv(map_file, sep='~')

  # Label columns
  df_pos.columns=['geneA symbol', 'geneA ID', 'geneB symbol', 'geneB ID', 'Cell line', 'PubmedID', 'Source', 'Score']
  df_neg.columns=['geneA symbol', 'geneA ID', 'geneB symbol', 'geneB ID', 'Cell line', 'PubmedID', 'Source', 'Score']
  df_funct.columns = ['GeneSym', 'Parent Function ID', 'Parent Function', 'Child1 Function ID', 'Child1 Function', 'Child2 Function ID', 'Child2 Function']
  return (df_pos, df_neg, df_funct)

def get_gene_list(df_funct):
  """
  Parses function information to get a list of all genes in the function mapping dataset to cross reference.
  params: df_funct: Pandas dataframe obtained from parse_data that contains function information of genes.
  return: list of genes
  """
  all_genes = []
  [all_genes.append(i) for i in df_funct['GeneSym'] if i not in all_genes]
  return all_genes 

def get_funct_list(df_funct):
  """
  Returns list of functions to be used as the features of the model.
  params: df_funct: Pandas dataframe obtained from parse_data that contains function information of genes.
  return: list of functions
  """
  functions = []
  [functions.append(i[1:]) for i in df_funct['Child1 Function'] if i[1:] not in functions]
  return functions
 

def select_genes(all_genes, df_pos, df_neg,
                 CL_desired:list=['K562;K562', 'K562', 'K562;Jurkat', 'K562;K562;K562', 'K562;K562;K562;K562'], 
                 S_desired:list=['GenomeRNAi', 'CRISPR/CRISPRi']):
  """
  Selects entries from the dataset that correspond to the desired cell line and source.
  params: CL_desired: cell-lines to pull from
          S_desired: desired source of data
          all_genes: list of genes to select from
  Returns: tuple of gene lists
  """
  pos_genes = []  # genes from Human_SL.csv that are also in KO dataset
  neg_genes = []  # genes from Human_nonSL.csv that are also in KO dataset

  pos_pairs = []  # List of gene pair tuples
  neg_pairs = []
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
  CLASS_WEIGHT = (len(pos_pairs)/(len(neg_pairs) + len(pos_pairs)))
  return (pos_genes, neg_genes, pos_pairs, neg_pairs)

def select_functions(df_funct, pos_genes, neg_genes):
  """
  Selects only functions that are associated with at least one gene in the given sets of gene pairs.
  params: df_funct: Pandas dataframe obtained from parse_data that contains function information of genes.
          pos_genes: list of genes from Human_SL.csv that are also in KO dataset as determined by 
          select_genes() function
          neg_genes: list of genes from Human_nonSL.csv that are also in KO dataset as determined by 
          select_genes() function
  return: dictionary of gene function associations.
  """
  funct_dict = {}
  for g, f in zip(df_funct['GeneSym'], df_funct['Child1 Function']):
    if g in pos_genes or g in neg_genes:
      #Add to function dictionary
      if g not in funct_dict.keys():
        funct_dict[g] = [f[1:]]
      else:
        funct_dict[g] = funct_dict[g] + [f[1:]]
  return funct_dict

def encode_pair(gene1, gene2, funct_dict, functions):
  """
  One-hot encodes a pair of genes. Helper function for encode().
  params: gene1: name of first gene in gene pair to encode
          gene2: name of second gene in gene pair to encode
          funct_dict: dictionary of gene function associations
          functions: list of functions to use as features
  return: list corresponding to the one-hot encoding of the pair.
  """
  g1_enc = [1 if f in funct_dict[gene1] else 0 for f in functions]
  g2_enc = [1 if f in funct_dict[gene2] else 0 for f in functions]
  return g1_enc + g2_enc

def encode(pos_pairs, neg_pairs, funct_dict, functions): 
  """
  One-hot encodes a set of gene pairs. Converts result to Pandas DataFrame which is then stored in a
  pickle file.
  params: pos_pairs: list of SL gene pairs obtained with select_genes()
          neg_pairs: list of non-SL gene pairs obtained with select_genes()
          funct_dict: dictionary of gene function associations to pass to encode_pair()
          functions: list of functions to use as features to pass to encode_pair()
  """
  data = []
  for (x,y) in pos_pairs:
    enc = encode_pair(x, y, funct_dict, functions)
    data.append([x]+[y]+enc+[1])
  for (x,y) in neg_pairs:
    enc = encode_pair(x, y, funct_dict, functions)
    data.append([x]+[y]+enc+[0])

  columns = [["Gene1"] + ["Gene2"] + ["Gene1 " + f for f in functions] + ["Gene2 " + f for f in functions] + ["SL"]]
  dataset = pd.DataFrame(data=data, columns=columns)

  file = open("data.p", "wb")
  pickle.dump(dataset, file)
  file.close()  

def split_data(filename:str='data.p'):
  """
  Split data into train and test sets. Store result in pickle file
  params: filename: name of file to load that contains a Pandas dataframe to split.  
  """
  with open(filename, 'rb') as file:
    data_table = pickle.load(file)
    df = pd.read_pickle("data.p")

  X = df.loc[:, df.columns.values[0:-1]].to_numpy()
  y = df.loc[:, 'SL'].to_numpy()

  # Split data into training and test sets 
  x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=SEED)

  # Convert to tensors
  x_train = torch.FloatTensor(x_train[:,2:].astype('float64'))
  y_train = torch.FloatTensor(y_train.astype('float64'))
  x_test = torch.FloatTensor(x_test[:,2:].astype('float64'))
  y_test = torch.FloatTensor(y_test.astype('float64'))
  # Save split data sets to a file
  with open('split_data.p', 'wb') as file:
    pickle.dump(x_train, file)
    pickle.dump(y_train, file)
    pickle.dump(x_test, file)
    pickle.dump(y_test, file)
    pickle.dump(CLASS_WEIGHT, file)

def main():
  # get data
  (df_pos, df_neg, df_funct) = parse_data()
  all_genes = get_gene_list(df_funct)
  funct_list = get_funct_list(df_funct)
  (pos_genes, neg_genes, pos_pairs, neg_pairs) = select_genes(all_genes, df_pos, df_neg)
  funct_dict = select_functions(df_funct, pos_genes, neg_genes)
  # perform one-hot encoding
  encode(pos_pairs, neg_pairs, funct_dict, funct_list)
  # generate training and test sets
  split_data()

if __name__ == "__main__":
  main()