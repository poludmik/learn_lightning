import json

import torch
from torch.utils.data import IterableDataset, DataLoader
import lightning as L
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)  # {input_ids, attention_mask, token_type_ids}

class TextDataset(IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path, 'r') as file:
            for line in file:
                # one line is json with "text" key, load it to a dict
                dict_line = json.loads(line)
                print(dict_line)
                # tokenize the text
                tokenized = tokenize_function(dict_line) # now convert to integers
                tokenized["input_ids"] = torch.Tensor(tokenized["input_ids"]).to(torch.int32)
                tokenized["attention_mask"] = torch.Tensor(tokenized["attention_mask"]).to(torch.int32)
                tokenized["token_type_ids"] = torch.Tensor(tokenized["token_type_ids"]).to(torch.int32)
                # print(tokenized)
                tokenized["label"] = torch.tensor(dict_line["label"])
                yield tokenized

    def __len__(self):
        with open(self.file_path, 'r') as file:
            return sum(1 for line in file)

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
            whole_dataset = TextDataset(self.data_dir)
            self.train_dataset = whole_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=8)

    def val_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1)


# large_text_dataset = LTDataModule("large_text_dataset/large_text_dataset.jsonl")
