# Script to fetch dataset

from datasets import load_dataset

data_name = "KisanVaani/agriculture-qa-english-only"
data = load_dataset(data_name, split = "train", cache_dir="data_cache")