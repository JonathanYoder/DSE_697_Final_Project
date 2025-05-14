# Fetch model tobe used on node without internet connection

import os
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

login(token='insert your token') #personal token removed for security purposes

# Select tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# save models to cache directory
model_path = "/gpfs/wolf2/olcf/trn040/scratch/kmn3/huggingface/meta-llama/Meta-Llama-3-8B"
tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)