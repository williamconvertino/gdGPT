import evaluation_base
from evaluation_base import GPT_CONFIG, GDM_CONFIG, CAUSAL_GDM_CONFIG, evaluate_model_with_config

config = CAUSAL_GDM_CONFIG
evaluate_model_with_config(config)