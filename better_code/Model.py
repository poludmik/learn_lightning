import torch
import lightning as L
from litgpt import Config
from litgpt.model import GPT
from litgpt.utils import chunked_cross_entropy
import math
from typing import Any, Optional
import signal
import os
# from deepspeed.ops.adam import FusedAdam
from torchmetrics.text import Perplexity


class LightningGemma2Module(L.LightningModule):
    def __init__(self, model_config: Config, hp_config: dict) -> None:
        super().__init__()
        self.model_config = model_config
        self.hp_config = hp_config
        self.requeue_flag = False
        self.module: Optional[torch.nn.Module] = None
        self.save_hyperparameters()

    def configure_model(self) -> None:
        self.module = GPT(self.model_config)
        state_dict = torch.load(self.hp_config["init_pth_model"])
        self.module.load_state_dict(state_dict)
        self.module.train()
        print("Loaded model")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.module.parameters(),
            lr=self.hp_config["peak_learning_rate"],
            weight_decay=self.hp_config["weight_decay"],
            betas=(self.hp_config["beta1"], self.hp_config["beta2"]),
            foreach=False,
        )

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        if self.requeue_flag:
            self.requeue_flag = False
            # send SIGUSR1 to requeue the job
            print("Sending SIGUSR1 to requeue job from the on_train_batch_start()", flush=True)
            os.kill(os.getpid(), signal.SIGUSR1)
            return

        if not self.hp_config["cosine_decay"]:
            return
        lr = get_lr(self.trainer.fit_loop.total_batch_idx, 
                    self.hp_config["peak_learning_rate"], 
                    self.hp_config["warmup_iters"], 
                    self.hp_config["lr_decay_iters"], 
                    self.hp_config["min_lr"]
                    )
        for optimizer in self.trainer.strategy.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
                self.log("current_lr", lr)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        torch.cuda.empty_cache()  # Call this periodically between batches or epochs (helps with memory fragmentation)
        input_ids = batch["input_ids"]
        targets = batch["labels"]
        # attention_mask = batch["attention_mask"]

        print(f"BATCH IDX: {batch_idx}", flush=True)

        logits = self.module(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0) # on full vectors (no chunking)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        input_ids = batch["input_ids"]
        targets = batch["labels"]
        # attention_mask = batch["attention_mask"]
        logits = self.module(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    # def on_validation_epoch_end(self):


    def setup_signal_handler(self):
        signal.signal(signal.SIGHUP, self.handle_sighup)

    def handle_sighup(self, signum, frame):
        print(f"Received SIGHUP signal in handle_sighup: {signum}")
        self.requeue_flag = True



# learning rate decay scheduler (cosine with warmup)
def get_lr(it, peak_learning_rate, warmup_iters, lr_decay_iters, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return peak_learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (peak_learning_rate - min_lr)
