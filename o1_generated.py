import os
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from itertools import chain


class LLMFinetuner(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        return optimizer


# Load your raw text dataset
datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

block_size = 1024  # Define the block size


# Tokenize the input texts
def tokenize_function(examples):
    return tokenizer(examples["text"])


tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=["text"])


# Group texts into blocks
def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples["input_ids"])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


lm_datasets = tokenized_datasets.map(group_texts, batched=True)

# Create DataLoaders
train_dataset = lm_datasets["train"]
eval_dataset = lm_datasets["validation"]

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=data_collator)
eval_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=data_collator)

# example of data collator:
print(next(iter(train_dataloader)))

# # Define the ModelCheckpoint callback
# checkpoint_callback = ModelCheckpoint(
#     every_n_train_steps=10000,
#     save_top_k=-1,
#     dirpath='checkpoints',
#     filename='model-{step}',
# )
#
# # Initialize the model
# model = LLMFinetuner()
#
# # Define the trainer
# trainer = L.Trainer(
#     max_epochs=3,
#     callbacks=[checkpoint_callback],
# )
#
# # Train the model
# trainer.fit(model, train_dataloader, eval_dataloader)
#
# # Save the fine-tuned model
# model.model.save_pretrained('fine_tuned_model')
