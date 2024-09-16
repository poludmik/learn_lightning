from datasets import load_dataset
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
from torch.utils.data import Dataset

### Another way to do it without using the datasets library:
import json

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def tokenize_and_save(input_file, output_file, tokenizer):
    token_ids = []
    for obj in read_jsonl(input_file):
        text = obj['text']  # Adjust the key according to your JSON structure
        tokens = tokenizer.encode(text, add_special_tokens=True)
        token_ids.extend(tokens)
    # Convert token_ids to a NumPy array
    arr = np.array(token_ids, dtype=np.uint32)  # Use np.uint32 if vocab size > 65535
    # Save to a binary file
    arr.tofile(output_file)

# Tokenize and save the dataset
tokenize_and_save('small_text_dataset.jsonl', 'binary_tokenized_dataset.bin', tokenizer)


class TokenizedDataset(Dataset):
    def __init__(self, data_file, block_size):
        self.data_file = data_file
        self.block_size = block_size
        self.data = np.memmap(self.data_file, dtype=np.uint32, mode='r')
        self.data_len = len(self.data) // self.block_size 
    
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
data_file = 'binary_tokenized_dataset.bin'  # Or 'tokenized_data.npy'
block_size = 5  # Adjust based on model's context length
batch_size = 3

# Choose Dataset type
dataset = TokenizedDataset(data_file, block_size)
print("Dataset_Length:", len(dataset), end="\n\n")  # Print the length of the dataset

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=1,
    # For IterableDataset, don't set shuffle=True
    # For Dataset, you can set shuffle=True
)

for idx, batch in enumerate(dataloader):
    for i in range(batch["input_ids"].shape[0]):
        print(tokenizer.decode(batch["input_ids"][i], skip_special_tokens=False))
        print(tokenizer.decode(batch["labels"][i], skip_special_tokens=False))
        print()
    
