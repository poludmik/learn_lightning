import math
from datasets import load_dataset
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')
from torch.utils.data import Dataset
import tqdm

### Another way to do it without using the datasets library:
import json

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def tokenize_and_save(input_file, output_file, tokenizer, max_length=1024, stride=512):
    token_ids = []
    for j, obj in tqdm.tqdm(enumerate(read_jsonl(input_file))):
        text = obj['text']
        encodings = tokenizer(text, truncation=False, add_special_tokens=True)
        input_ids = encodings['input_ids']
        for i in range(0, len(input_ids), stride):
            chunk = input_ids[i:i + max_length]
            if len(chunk) < max_length:
                break
            token_ids.extend(chunk)
        
    arr = np.array(token_ids, dtype=np.uint32)
    arr.tofile(output_file)

# Tokenize and save the dataset
# tokenize_and_save('czech_news_dataset_v2.jsonl', 'czech_news_dataset_full_v2.bin', tokenizer)


class TokenizedDataset(Dataset):
    def __init__(self, data_file, block_size):
        self.data_file = data_file
        self.block_size = block_size
        self.data = np.memmap(self.data_file, dtype=np.uint32, mode='r')
        self.data_len = math.ceil(len(self.data) / self.block_size)
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx*self.block_size:idx*self.block_size+self.block_size].astype(np.uint32))
        y = torch.from_numpy(self.data[idx*self.block_size+1:idx*self.block_size+1+self.block_size].astype(np.uint32))
        # pad the last sequence if it's shorter than block_size
        if y.size(0) < self.block_size:
            pad = torch.zeros(self.block_size - y.size(0), dtype=torch.uint32)
            y = torch.cat([y, pad])
        # print({'input_ids': x, 'labels': y})
        return {'input_ids': x, 'labels': y}

from torch.utils.data import DataLoader

# Parameters
data_file = 'czech_news_dataset_full_v2.bin'  # Or 'tokenized_data.npy'
block_size = 50  # Adjust based on model's context length
batch_size = 1

# Choose Dataset type
dataset = TokenizedDataset(data_file, block_size)
print("Dataset_Length:", len(dataset), end="\n\n")  # Print the length of the dataset

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=1,
    shuffle=True,
)

for idx, batch in enumerate(dataloader):
    for i in range(batch["input_ids"].shape[0]):
        print(tokenizer.decode(batch["input_ids"][i], skip_special_tokens=False))
        print(tokenizer.decode(batch["labels"][i], skip_special_tokens=False))
        print()
    break
