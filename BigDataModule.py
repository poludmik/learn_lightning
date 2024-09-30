from datetime import datetime
import math
import os
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import numpy as np

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


class MyDataModule(L.LightningDataModule):
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
        val_size = len(full_dataset) - train_size

        # Split the dataset into training and validation sets
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

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
