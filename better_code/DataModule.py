from datetime import datetime
import math
import os
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import numpy as np
import random

random.seed(228)
torch.manual_seed(228) # to ensure reproducibility for the dataloader

class TokenizedDataset(Dataset):
    def __init__(self, data_file, block_size):
        self.data_file = data_file
        self.block_size = block_size
        self.data = np.memmap(self.data_file, dtype=np.uint32, mode='r')  # Keep as read-only
        self.data_len = math.ceil(len(self.data) / self.block_size)
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        x_data = self.data[start:end]
        y_data = self.data[start + 1:end + 1]

        # Copy the arrays to make them writable
        x = torch.from_numpy(x_data.copy()).to(torch.long)
        y = torch.from_numpy(y_data.copy()).to(torch.long)

        # Pad x if necessary
        if x.size(0) < self.block_size:
            pad_size = self.block_size - x.size(0)
            pad = torch.zeros(pad_size, dtype=torch.long)
            x = torch.cat([x, pad], dim=0)
            # print red:
            print("\033[91m" + f"PADDED x!" + "\033[0m")

        # Pad y if necessary
        if y.size(0) < self.block_size:
            pad_size = self.block_size - y.size(0)
            pad = torch.zeros(pad_size, dtype=torch.long)
            y = torch.cat([y, pad], dim=0)
            print("\033[91m" + f"PADDED y!" + "\033[0m")
        
        attention_mask = (x != 0).long()

        return {'input_ids': x, 'labels': y, 'attention_mask': attention_mask}


class Gemma2DataModule(L.LightningDataModule):
    def __init__(self, data_file, block_size, batch_size, num_workers):
        super().__init__()
        self.data_file = data_file
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Initialize the full dataset
        full_dataset = TokenizedDataset(self.data_file, self.block_size)

        # Calculate split sizes
        train_size = int(0.999 * len(full_dataset))
        # train_size = int(len(full_dataset) - 1)
        val_size = len(full_dataset) - train_size

        print(">>>>>>>>>>>>>>>>>>>>>>>>")
        print(f"full_dataset len: {len(full_dataset)}")
        print(f"train_size: {train_size}")
        print(f"val_size: {val_size}")
        print(">>>>>>>>>>>>>>>>>>>>>>>>")

        # Split the dataset into training and validation sets
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

        print("#"*20)
        print(f"train_dataset len: {len(self.train_dataset)}")
        print(f"val_dataset len: {len(self.val_dataset)}")
        print("#"*20)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
