from typing import Optional
from dataclasses import dataclass

@dataclass
class Config:
  
  model_type: str
  
  context_size: int
  vocab_size: int
  
  d_embed: int
  n_layer: int
  n_head: int
  
  dropout: float = 0.1
  
  attn_kernel_fn: str = 'softmax'
  use_ff: bool = True
  end_ff: bool = False
  use_ppe: bool = False
  use_nto: bool = False
  use_gd_bias: bool = False

  use_skip=False
  
  def __post_init__(self):
    assert self.model_type in ['GPT', 'GDM', 'PGD', 'CausalGDM', 'CausalGPT']
    assert self.attn_kernel_fn in ['softmax', 'linear', 'rbf', 'laplacian']
    self.d_ff = self.d_embed * 4
    self.name = f'{self.model_type}_{self.d_embed}D_{self.n_head}H_{self.n_layer}L_K={self.attn_kernel_fn}'
    if self.use_ff:
      self.name += '_FF'
    if self.use_ppe:
      self.name += '_PPE'
    if self.use_nto:
      self.name += '_NTO'
    if self.model_type == 'GDM' and self.use_gd_bias:
      self.name += '_GDB'
    if not self.use_skip:
      self.name += '_NS'
    if self.end_ff:
      self.name += '_EFF'
    if self.model_type == 'CausalGDM' and self.use_skip:
      self.name += '_SKIP'