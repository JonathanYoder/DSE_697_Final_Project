# Run inference on a model
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

torch.cuda.empty_cache()

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# load model from saved checkpoint
#model_path = "/gpfs/wolf2/olcf/trn040/scratch/kmn3/huggingface/meta-llama/Meta-Llama-3-8B"
#tokenizer = AutoTokenizer.from_pretrained(model_path)
#model = AutoModelForCausalLM.from_pretrained(model_path)

#pipeline = pipeline(task="text-generation", model="llama-3-ag-qa-tuned", tokenizer=tokenizer, device=0)
pipeline = pipeline(task="text-generation", model="TennAG_model_1ep", device=0)
output = pipeline("How is wheat grown?")
print(output)