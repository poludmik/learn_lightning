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
# from litdata import StreamingDataset, StreamingDataLoader, TokensLoader
# from tqdm import tqdm
from lightning.pytorch.plugins.environments import SLURMEnvironment
import signal
from BigDataModule import TokenizedDataset, MyDataModule

L.seed_everything(228, workers=True)


# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

# Parameters
# data_file = 'large_text_dataset/czech_news_dataset_full_v2.bin'  # Update path as needed
data_file = "../dataset-playground/azure_data/czech_llm_data/czech-llm-dataset-complete/merged_all_files.bin"
block_size = 1024  # Adjust based on model's context length
batch_size = 20
num_workers = 15
max_epochs = 1  # Set desired number of epochs

# Initialize the DataModule
data_module = MyDataModule(
    data_file=data_file,
    block_size=block_size,
    batch_size=batch_size,
    num_workers=num_workers,
)

data_module.setup()

# print("Dataset_Length:", len(dataset), end="\n\n")  # total number of instances

# dataloader = DataLoader(
#     dataset,
#     batch_size=batch_size,
#     num_workers=15,
#     shuffle=True,
# )

# dataset = StreamingDataset(
#   input_dir=f"./litdata/optimized_dataset_mikhail",
#   item_loader=TokensLoader(block_size=400),
#   shuffle=True,
#   drop_last=True,
# )
# dataloader = StreamingDataLoader(dataset, batch_size=8, pin_memory=True, num_workers=os.cpu_count())


class GPT2Finetuner(L.LightningModule):
    def __init__(self, warmup_steps=0, total_steps=0):
        super().__init__()

        if warmup_steps == 0:
            print("\033[91m" + "Warning: warmup_steps is set to 0." + "\033[0m")

        if total_steps == 0:
            print("\033[91m" + "Warning: total_steps is set to 0." + "\033[0m")

        self.save_hyperparameters()  # Saves hyperparameters for checkpointing

        self.gpt2 = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        self.gpt2.train()

    def training_step(self, batch, batch_idx):
        targets = batch["labels"]
        outputs = self.gpt2(batch["input_ids"])

        print("Batch idx:", batch_idx) # to track the progress if resumed from a checkpoint (progress bar will show from 0)

        # Compute loss using chunked cross-entropy for efficiency
        loss = litgpt.utils.chunked_cross_entropy(outputs.logits, targets)
        self.log("train_loss", loss, prog_bar=True)

        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("current_lr", current_lr)

        return loss
    
    def validation_step(self, batch, batch_idx):
        targets = batch["labels"]
        outputs = self.gpt2(batch["input_ids"])
        loss = litgpt.utils.chunked_cross_entropy(outputs.logits, targets)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
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
wandb_logger = WandbLogger(project="learn-lightning", name="GPT-2 at " + current_time)

# Calculate total steps
num_training_batches = len(data_module.train_dataloader())
total_steps = max_epochs * num_training_batches
warmup_steps = int(0.01 * total_steps)  # 1% of total steps

model = GPT2Finetuner(warmup_steps=warmup_steps, total_steps=total_steps)
wandb_logger.watch(model, log="parameters", log_graph=False)

# checkpoint_path = "lightning_logs/version_1706648/checkpoints/epoch=0-step=50791.ckpt"
# model = GPT2Finetuner.load_from_checkpoint(checkpoint_path, warmup_steps=warmup_steps, total_steps=total_steps)
# model.train()

checkpoint_callback = ModelCheckpoint(
        dirpath="my_gpt2_big_checkpoints",
        filename="cp-{epoch:1d}-{step:02d}",
        every_n_train_steps=10000,
        save_top_k=-1  # keep all checkpoints!
    )

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
    callbacks=[checkpoint_callback, 
            #    progress_bar # doesn't work in slurm output files
            ],
    devices=7,
    strategy="ddp",
    val_check_interval=2000, # every N steps, check validation. Or set to 0.25 to check every 25% of 1 epoch
    # limit_train_batches=7.42597946385483e-04,
    # overfit_batches=0.00001,
    deterministic=True,
    plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)]
)

#Start training
trainer.fit(
    model,
    # train_dataloaders=dataloader,
    datamodule=data_module,
    # ckpt_path="my_gpt2_big_checkpoints/cp-epoch=0-step=100.ckpt", # if resuming with 1 epoch limit, it will start from the number of batches it has already seen and stop when it sees everything. that is, now it will take total_num_batches - already_seen_batches steps and end.
)
torch.save(model.state_dict(), 'pth_models/model_gpt2_big.pth')


# # # run model for prediction:
# model = GPT2Finetuner()

# # load from the checkpoint "my_gpt2_checkpoints/cp-epoch=0-step=4000.ckpt"
# model.load_state_dict(torch.load('my_gpt2_checkpoints/cp-epoch=0-step=4000.ckpt')['state_dict'])

# # model.load_state_dict(torch.load('model_gpt2_cznews.pth', weights_only=True))
# model.eval()
# input_text = """Národní muzeum se nachází """
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
