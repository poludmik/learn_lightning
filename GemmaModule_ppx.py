from datetime import datetime
from datasets import load_dataset
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import lightning as L
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR, CosineAnnealingLR
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
from torchmetrics.text import Perplexity


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

    def training_step(self, batch, batch_idx):
        targets = batch["labels"]
        outputs = self.gemma2(batch["input_ids"], attention_mask=batch["attention_mask"])

        print(f"BATCH IDX: {batch_idx}")

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

        pad_token_id = 0 # From Gemma 2 tokenizer config
        logits = outputs.logits
        perplexity = Perplexity(ignore_index=None).to(torch.device('cuda'))
        score = perplexity(preds=logits, target=targets)
        self.log("val_perplexity_pad", score, prog_bar=True, sync_dist=True)
        
        perplexity = Perplexity(ignore_index=pad_token_id).to(torch.device('cuda'))
        score = perplexity(preds=logits, target=targets)
        self.log("val_perplexity", score, prog_bar=True, sync_dist=True)
        self.log("val_loss_exp", loss.exp(), prog_bar=True, sync_dist=True)
        

        return loss

    def configure_optimizers(self):
        optimizer = FusedAdam(self.parameters(), lr=2e-5)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)

        total_steps = self.hparams.total_steps
        warmup_steps = self.hparams.warmup_steps
        remaining_steps = total_steps - warmup_steps  # Steps after warmup for cosine decay
        print(f"/////////GemmaModule//////////////")
        print(f"Total steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
        print(f"Remaining steps: {remaining_steps}")
        print(f"/////////GemmaModule//////////////")

        # Warmup scheduler (linear warmup)
        warmup_scheduler = LinearLR(
            optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps
        )

        # Cosine annealing scheduler (after warmup)
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=remaining_steps, eta_min=1e-6  # Min LR at the end of cosine decay
        )

        # Sequential scheduler to combine warmup and cosine decay
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
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
