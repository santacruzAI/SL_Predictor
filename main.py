
import sys
import torch
from model import Model
from Preprocessing import *

def get_data():
  pass

def tune_model(model, lr:list, epochs:list):
  pass

def predict(path):
  """
  params: path: filepath of .pth trained model
  """
  model = Model()
  model.load_state_dict(torch.load(path))

  g1 = input("Please enter the first gene: ")
  # TODO: check that g1 is valid  *********
  g2 = input("Please enter the second gene: ")
  # TODO: check that g2 is valid  *********
  enc = encode_pair(g1, g2)
  pred = model.predict_one(enc)
  if pred == 1:
    print("The model predicts that {} and {} form a synthetic lethal pair.".format(g1,g2))
  else:
    print("The model predicts that {} and {} do NOT form a synthetic lethal pair.".format(g1,g2))

def make_visuals(model):
  pass
  
if __name__ == "__main__":
  args = sys.argv[1:]
  
  # TODO parse args
