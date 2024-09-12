import os
import torch
from torch import nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data
import lightning as L

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import evaluate
import numpy as np
from datasets import load_dataset
metric = evaluate.load("accuracy")



class BertMNLIFinetuner(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.bert = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
        self.bert.train()

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        input_ids = torch.cat(input_ids, dim=0)
        # make one dimension for batch
        input_ids = input_ids.view(-1, input_ids.size(0))

        attention_mask = batch["attention_mask"]
        attention_mask = torch.cat(attention_mask, dim=0)
        attention_mask = attention_mask.view(-1, attention_mask.size(0))

        # forward pass
        logits = self.bert(input_ids, attention_mask)
        # print(logits.logits.shape) # [1, 5]

        # calculate loss
        loss = F.cross_entropy(logits.logits, batch["label"])
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        input_ids = torch.cat(input_ids, dim=0)
        # make one dimension for batch
        input_ids = input_ids.view(-1, input_ids.size(0))

        attention_mask = batch["attention_mask"]
        attention_mask = torch.cat(attention_mask, dim=0)
        attention_mask = attention_mask.view(-1, attention_mask.size(0))

        # forward pass
        logits = self.bert(input_ids, attention_mask)
        # print(logits.logits.shape) # [1, 5]

        # calculate loss
        loss = F.cross_entropy(logits.logits, batch["label"])
        self.log("val_loss", loss)
        print("Validation loss: ", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-6)
        return optimizer


tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)  # {input_ids, attention_mask, token_type_ids}

if tokenizer:
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    dataset = load_dataset("yelp_review_full")
    # print(dataset["train"][100])

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    model = BertMNLIFinetuner()
    trainer = L.Trainer(
        max_epochs=3,
        # callbacks=[checkpoint_callback]
    )

    trainer.fit(model, DataLoader(small_train_dataset, batch_size=1), DataLoader(small_eval_dataset, batch_size=1))
    torch.save(model.state_dict(), 'model_bert.pth')
else:
    model = BertMNLIFinetuner()
    model.load_state_dict(torch.load('model_bert.pth'))
    model.eval()

    text = "The restaurant was great."

    inputs = tokenizer(text, return_tensors="pt")
    logits = model.bert(**inputs)
    print(logits)
    print(torch.argmax(logits.logits, dim=-1))




