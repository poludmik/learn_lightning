from datetime import datetime
from datasets import load_dataset
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import lightning as L
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR
from lightning.pytorch.callbacks import ModelCheckpoint
import litgpt  # Ensure litgpt is installed and accessible
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.callbacks.progress import RichProgressBar
# from litdata import StreamingDataset, StreamingDataLoader, TokensLoader
# from tqdm import tqdm
from lightning.pytorch.plugins.environments import SLURMEnvironment
import signal
from BigDataModule import TokenizedDataset, MyDataModule
from deepspeed.ops.adam import FusedAdam


class Gemma2Finetuner(L.LightningModule):
    def __init__(self, warmup_steps=0, total_steps=0):
        super().__init__()

        if warmup_steps == 0:
            print("\033[91m" + "Warning: warmup_steps is set to 0." + "\033[0m")

        if total_steps == 0:
            print("\033[91m" + "Warning: total_steps is set to 0." + "\033[0m")

        self.save_hyperparameters()  # Saves hyperparameters for checkpointing
        self.gemma2 = None

        self.gemma2 = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b",
                                                            attn_implementation='eager',
                                                            )
        self.gemma2.train()


    # def configure_model(self): # efficient way to load the model (not for deepspeed)
    #     if self.gemma2 is not None:
    #         return
    #     self.gemma2 = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b",
    #                                                         attn_implementation='eager',
    #                                                         )
    #     self.gemma2.train()

    def training_step(self, batch, batch_idx):
        targets = batch["labels"]
        outputs = self.gemma2(batch["input_ids"])

        # check sizes of outputs and targets first two dimensions of the shapes
        if outputs.logits.shape[0] != targets.shape[0] or outputs.logits.shape[1] != targets.shape[1]:
            print("      >>>>> Shapes of outputs and targets do not match: " + str(outputs.logits.shape) + " vs " + str(targets.shape) + "<<<<<")
        
        print("Batch idx:", batch_idx) # to track the progress if resumed from a checkpoint (progress bar will show from 0)
        
        # Compute loss using chunked cross-entropy for efficiency
        loss = litgpt.utils.chunked_cross_entropy(outputs.logits, targets)
        self.log("train_loss", loss, prog_bar=True)
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("current_lr", current_lr)
        
        return loss

    def validation_step(self, batch, batch_idx):
        targets = batch["labels"]
        outputs = self.gemma2(batch["input_ids"])
        loss = litgpt.utils.chunked_cross_entropy(outputs.logits, targets)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = FusedAdam(self.parameters(), lr=3e-5)

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