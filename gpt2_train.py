from datetime import datetime
import math
import os
from datasets import load_dataset
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import lightning as L
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR
from lightning.pytorch.callbacks import ModelCheckpoint
import litgpt  # Ensure litgpt is installed and accessible
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.callbacks.progress import RichProgressBar


# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

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

# Parameters
data_file = 'large_text_dataset/czech_news_dataset_full_v2.bin'  # Update path as needed
block_size = 1024  # Adjust based on model's context length
batch_size = 16
max_epochs = 1  # Set desired number of epochs

# Create Dataset and DataLoader
dataset = TokenizedDataset(data_file, block_size)

print("Dataset_Length:", len(dataset), end="\n\n")  # Print the length of the dataset

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=1,
    shuffle=True,
)

class GPT2Finetuner(L.LightningModule):
    def __init__(self, warmup_steps, total_steps):
        super().__init__()
        self.save_hyperparameters()  # Saves hyperparameters for checkpointing

        # Initialize the GPT-2 model
        self.gpt2 = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        self.gpt2.train()

    def training_step(self, batch, batch_idx):
        targets = batch["labels"]
        outputs = self.gpt2(batch["input_ids"])

        # Compute loss using chunked cross-entropy for efficiency
        loss = litgpt.utils.chunked_cross_entropy(outputs.logits, targets)
        self.log("train_loss", loss, prog_bar=True)

        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("current_lr", current_lr)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-5)

        total_steps = self.hparams.total_steps
        warmup_steps = self.hparams.warmup_steps

        warmup_scheduler = LinearLR(
            optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps
        )

        constant_scheduler = ConstantLR(
            optimizer, factor=1.0, total_iters=total_steps - warmup_steps
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, constant_scheduler],
            milestones=[warmup_steps]
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',       # Update the scheduler after each step
                'frequency': 1,
            }
        }

current_time = datetime.now().strftime("%d.%m._%H:%M")
wandb_logger = WandbLogger(project="learn-lightning", 
                           name="GPT-2 at " + current_time, 
                           )


# Calculate total_steps and warmup_steps
total_steps = max_epochs * len(dataloader)
warmup_steps = int(0.01 * total_steps)  # 10% of total steps

# Initialize the model with warmup_steps and total_steps
model = GPT2Finetuner(warmup_steps=warmup_steps, total_steps=total_steps)
wandb_logger.watch(model, log="parameters", log_graph=False)

checkpoint_path = "lightning_logs/version_1706648/checkpoints/epoch=0-step=50791.ckpt"
# model = GPT2Finetuner.load_from_checkpoint(checkpoint_path, warmup_steps=warmup_steps, total_steps=total_steps)
# model.train()

checkpoint_callback = ModelCheckpoint(
        dirpath="my_gpt2_checkpoints",
        filename="cp-{epoch:1d}-{step:02d}",
        every_n_train_steps=2000,
        save_top_k=-1  # keep all checkpoints!
    )

# create your own theme!
progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="chartreuse1",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="misty_rose1",
        processing_speed="medium_purple2",
        metrics="hot_pink2",
        metrics_text_delimiter="\n",
        metrics_format=".3",
    )
)

# Initialize the Trainer
trainer = L.Trainer(
    max_epochs=max_epochs,
    # Add any additional Trainer arguments here (e.g., gpus, callbacks)
    logger=wandb_logger,
    callbacks=[checkpoint_callback, progress_bar]
)

# Start training
trainer.fit(
    model,
    train_dataloaders=dataloader,
    # Optionally, add validation dataloaders if available
)

# Save the fine-tuned model
torch.save(model.state_dict(), 'model_gpt2_cznews_ctnd.pth')


# # run model for prediction:
# model = GPT2Finetuner()
# model.load_state_dict(torch.load('model_gpt2_cznews.pth', weights_only=True))
# model.eval()
# input_text = """Na otázku odpověděl: """
# input_ids = tokenizer.encode(input_text, return_tensors='pt')

# attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

# output = model.gpt2.generate(input_ids, attention_mask=attention_mask, 
#                              max_new_tokens=30, 
#                              num_return_sequences=1, 
#                              no_repeat_ngram_size=2, 
#                              top_k=50,
#                             #  top_p=0.95, 
#                             #  do_sample=False,
#                             #  temperature=0.7
#                              )
# print(">>>>>>")
# print(tokenizer.decode(output[0], skip_special_tokens=False))
