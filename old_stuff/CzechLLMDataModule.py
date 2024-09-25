import json
import torch
from torch.utils.data import DataLoader #, IterableDataset
import lightning as L
from torch.utils.data import random_split, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import math
import math
from datasets import load_dataset
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import lightning as L


class TokenizedDataset(Dataset):
    def __init__(self, data_file, block_size):
        self.data_file = data_file
        self.block_size = block_size
        self.data = np.memmap(self.data_file, dtype=np.uint32, mode='r')
        self.data_len = math.ceil(len(self.data) / self.block_size)
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        x = torch.from_numpy(
            self.data[idx*self.block_size : idx*self.block_size + self.block_size]
            .astype(np.uint32)
        ).to(torch.long)
        y = torch.from_numpy(
            self.data[idx*self.block_size + 1 : idx*self.block_size + 1 + self.block_size]
            .astype(np.uint32)
        ).to(torch.long)
        # Pad the last sequence if it's shorter than block_size
        if y.size(0) < self.block_size:
            pad = torch.zeros(self.block_size - y.size(0), dtype=torch.long)
            y = torch.cat([y, pad])
        return {'input_ids': x, 'labels': y}

# TODO: Implement the class
class LTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = None
        self.train_dataset = None
        self.val_dataset = None # if would need a val dataset, then need a separate data file and it's dataset, because you can't split an IterableDataset
        self.test_dataset = None

    def prepare_data(self):
        # download (if no directory exists), from Azure/s3, etc
        # tokenize, save to disk
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloader

        if stage == "fit":
            whole_dataset = TokenizedDataset(self.data_dir)
            self.train_dataset = whole_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=8)

    def val_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1)


# large_text_dataset = LTDataModule("large_text_dataset/large_text_dataset.jsonl")
