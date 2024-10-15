# Code from Lit-GPT https://github.com/Lightning-AI/lit-gpt
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import FSDPStrategy

from BigDataModule import MyDataModule
from lightning.pytorch.loggers import WandbLogger
from datetime import datetime

from litgpt import Config
from litgpt.model import GPT, Block
# from litgpt.speed_monitor import SpeedMonitorCallback
from litgpt.utils import chunked_cross_entropy, get_default_supported_precision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import BackwardPrefetch
import functools

model_name = "google/gemma-2-2b"
name = "lit-openwebtext"
out_dir = Path("out") / name
data_dir = Path("/data/aniket/openwebtext/") / name
save_interval = 1000
eval_interval = 1000
eval_iters = 100
log_interval = 1

# Hyperparameters
learning_rate = 2e-5
block_size = 8192
accumulate_batch_size = 1
micro_batch_size = 1
gradient_accumulation_steps = accumulate_batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
max_iters = 600000  # num_epochs * (epoch_size // micro_batch_size) // devices
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
decay_lr = True
warmup_iters = 100
lr_decay_iters = 400
min_lr = 6e-6

hparams = {
    k: v
    for k, v in locals().items()
    if isinstance(v, (int, float, str)) and not k.startswith("_")
}

class LightningGPTModule(L.LightningModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.module: Optional[torch.nn.Module] = None
        self.save_hyperparameters()

    def configure_model(self) -> None:
        self.module = GPT(self.config)
        state_dict = torch.load("checkpoints/google/gemma-2-2b/lit_model.pth")
        self.module.load_state_dict(state_dict)
        print("Loaded model")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.module.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            foreach=False,
        )

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        if not decay_lr:
            return
        lr = get_lr(self.trainer.fit_loop.total_batch_idx)
        for optimizer in self.trainer.strategy.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
                self.log("current_lr", lr)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        torch.cuda.empty_cache()  # Call this periodically between batches or epochs (helps with memory fragmentation)
        input_ids = batch["input_ids"]
        targets = batch["labels"]
        logits = self.module(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        input_ids = batch["input_ids"]
        targets = batch["labels"]
        logits = self.module(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

def main(
    devices: int = 1,
    precision: Optional[str] = None,
) -> None:
    precision = precision or get_default_supported_precision(training=True)
    print(f"Using precision: {precision}")

    strategy = FSDPStrategy(
        auto_wrap_policy={Block},
        activation_checkpointing_policy={Block},
        # limit_all_gathers=True,
        # forward_prefetch=True,  # Enable forward prefetch for overlapping forward passes
        # sharding_strategy="HYBRID_SHARD", # TODO: try on 2 nodes with (2, 8)?
        # device_mesh=(4, 2),
        cpu_offload=False,
    )

    model_checkpoint = ModelCheckpoint(
        dirpath=out_dir, every_n_train_steps=save_interval, save_last=True, verbose=True
    )

    config = Config.from_name(model_name)
    t0 = time.perf_counter()
    model = LightningGPTModule(config)

    current_time = datetime.now().strftime("%d.%m._%H:%M")
    wandb_logger = WandbLogger(project="learn-lightning", name="Gemma2 at " + current_time)
    wandb_logger.watch(model, log="parameters", log_graph=False)

    trainer = L.Trainer(
        num_nodes=1,
        devices=8,
        strategy=strategy,
        precision=precision,
        logger=wandb_logger,
        callbacks=[model_checkpoint],
        max_steps=max_iters,
        # max_epochs=1,
        limit_val_batches=eval_iters,
        accumulate_grad_batches=gradient_accumulation_steps,
        log_every_n_steps=log_interval,
        val_check_interval=eval_interval,
    )

    L.seed_everything(
        1337, workers=True
    )  # same seed for every process to init model (FSDP)

    trainer.print(hparams)

    if trainer.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    trainer.print(
        f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds."
    )

    data_file = "../dataset-playground/azure_data/czech_llm_data/czech-llm-dataset-complete/cswiki_only.bin"

    # Initialize the DataModule
    data_module = MyDataModule(
        data_file=data_file,
        block_size=block_size,
        batch_size=micro_batch_size,
        num_workers=9,
    )
    data_module.setup()

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    t0 = time.perf_counter()
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path="last")
    trainer.print(f"Training time: {(time.perf_counter()-t0):.2f}s")
    if trainer.strategy.root_device.type == "cuda":
        trainer.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    # torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(main)