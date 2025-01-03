import os
from datasets import load_dataset

from src.tokenizers import Tokenizer

HUGGINGFACE_PATH = 'ajibawa-2023/Children-Stories-Collection'

TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/tokenizers/cs_tokenizer_15k')
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/datasets/ChildrenStories')

class ChildrenStoriesTokenizer(Tokenizer):
  
  def __init__(self):
    
    self.name = 'cs_tokenizer_15k'
    
    if not os.path.exists(TOKENIZER_DIR):
      print('Creating CS tokenizer files...')
      dataset = load_dataset(HUGGINGFACE_PATH, cache_dir=f'{DATASET_DIR}/raw')
      Tokenizer.generate_tokenizer_files(dataset['train']['text'], TOKENIZER_DIR, vocab_size=15000)
  
    super().__init__(TOKENIZER_DIR)