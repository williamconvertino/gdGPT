import os
import torch

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../checkpoints')

def get_latest_epoch_path(model_name, max_epoch=None):

  model_files = [
    f for f in os.listdir(MODELS_DIR) 
    if f.startswith(f"{model_name}_epoch_") and f.endswith(".pt")
  ]
  
  if not model_files:
    raise FileNotFoundError(f"No model checkpoints found for {model_name} in {MODELS_DIR}")

  model_epochs = []
  
  for file in model_files:
    epoch = int(file.split("_epoch_")[1].split(".pt")[0])
    if max_epoch is None or epoch <= max_epoch:
      model_epochs.append((epoch, file))
    
  if not model_epochs:
    raise FileNotFoundError(f"No valid model checkpoints found for {model_name} with max_epoch={max_epoch}")

  latest_epoch, latest_model_file = max(model_epochs, key=lambda x: x[0])

  checkpoint_path = os.path.join(MODELS_DIR, latest_model_file)
  
  return checkpoint_path, latest_epoch

def load_mrm(model, max_epoch=None):
  
  model_name = model.name
  
  model_path, latest_epoch = get_latest_epoch_path(model_name, max_epoch)
  
  model.load_state_dict(torch.load(model_path, weights_only=True))

  print(f"Loaded model with epoch={latest_epoch}")
  return model