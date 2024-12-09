import os
import sys
import subprocess

# Add parent directory to path

path = os.path.abspath(os.path.join(os.getcwd(), '../'))
if path not in sys.path:
  sys.path.append(path)
  
# Install requirements
    
try:
  subprocess.check_call(["pip", "install", "-r", "../requirements.txt", "--quiet"])
  print("Requirements installed")
except:
  print("Failed to install requirements")

# Basic experiment setup

import torch
from src.models import CausalGDM, GPT, GDM, Config, CausalGDM
from src.training import train_model
from src.datasets import TinyStoriesDataset
from src.tokenizers import TinyStoriesTokenizer
from src.util import load_mrm
from src.evaluation import evaluate_model_generation

DEFAULT_VOCAB_SIZE = 10002
  
GPT_CONFIG = Config(
  
  model_type='GPT',
  
  context_size=256,
  vocab_size=DEFAULT_VOCAB_SIZE,
  
  d_embed=512,
  n_layer=1,
  n_head=8,
  
  dropout=0.1,
  
  attn_kernel_fn='softmax',
  
  use_ff=True,
  use_ppe=False
)

GDM_CONFIG = Config(
  
  model_type='GDM',
  
  context_size=256,
  vocab_size=DEFAULT_VOCAB_SIZE,
  
  d_embed=512,
  n_layer=1,
  n_head=8,
  
  dropout=0.1,
  
  attn_kernel_fn='softmax',
  
  use_ff=False,
  use_ppe=False,
  use_nto=True
)

CAUSAL_GDM_CONFIG = Config(
  model_type='CausalGDM',
  
  context_size=256,
  vocab_size=DEFAULT_VOCAB_SIZE,
  
  d_embed=512,
  n_layer=1,
  n_head=8,
  
  dropout=0.1,
  
  use_ff=False
)
  
def evaluate_model_with_config(config, max_epoch=None):
  
  if config.model_type == "GPT":
    model = GPT(config)
  elif config.model_type == "GDM":
    model = GDM(config)
  elif config.model_type == "CausalGDM":
    model = CausalGDM(config)
  else:
    raise ValueError("Invalid model type")

  model = load_mrm(model, max_epoch=max_epoch)

  tokenizer = TinyStoriesTokenizer()
  val_dataset = TinyStoriesDataset(tokenizer, 'val', context_size=config.context_size)
  
  evaluate_model_generation(model, tokenizer, val_dataset)
  