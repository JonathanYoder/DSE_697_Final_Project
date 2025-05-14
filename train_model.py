# Code to train model

import os
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Load Model
model_path = "/gpfs/wolf2/olcf/trn040/scratch/kmn3/huggingface/meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
base_model = AutoModelForCausalLM.from_pretrained(model_path)

# Load Data
training_data = load_dataset('/gpfs/wolf2/olcf/trn040/scratch/kmn3/data_cache', split = "train")
#training_data = training_data.map(remove_columns=["question","answers"])
print(training_data.shape)
print(training_data[11])

# Training Params
train_params = SFTConfig(
    output_dir="./results_modified",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    save_steps=50,
    logging_steps=50,
    learning_rate=4e-5,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    dataset_text_field="question",
    label_names="answers"
)

from peft import get_peft_model
# LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=8,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, peft_parameters)
model.print_trainable_parameters()

# Trainer with LoRA configuration
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=training_data,
    peft_config=peft_parameters,
    processing_class=tokenizer,
    args=train_params
)

# Training
fine_tuning.train()
# Save model
model.save_pretrained("/gpfs/wolf2/olcf/trn040/scratch/kmn3/TennAG_model_1ep")
