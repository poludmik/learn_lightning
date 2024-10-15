import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import FSDPStrategy
from datetime import datetime

from litgpt import Config
from litgpt.model import Block
from litgpt.utils import get_default_supported_precision

import yaml
import signal
from lightning.pytorch.plugins.environments import SLURMEnvironment
from Model import LightningGemma2Module
from DataModule import Gemma2DataModule

from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.callbacks.progress import RichProgressBar


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return convert_values(config_dict)

def convert_values(config: dict) -> dict:
    for key, value in config.items():
        try:
            float_value = float(value)
            
            # If the float value is actually an integer, convert it to int
            if float_value.is_integer():
                config[key] = int(float_value)
            else:
                config[key] = float_value
        except ValueError:
            pass # leave as string
    return config


L.seed_everything(228, workers=True)


def main(hp_config_path: str) -> None:
    hp_config = load_yaml_config(hp_config_path) # dict

    model = LightningGemma2Module(model_config=Config.from_name(hp_config["model_name"]), 
                                    hp_config=hp_config)
    
    model.setup_signal_handler()

    data_module = Gemma2DataModule(
        data_file=hp_config["training_data_path"],
        block_size=hp_config["block_size"],
        batch_size=hp_config["micro_batch_size"],
        num_workers=9,
    )
    data_module.setup()

    precision = get_default_supported_precision(training=True)
    print(f"Using precision: \033[94m{precision}\033[0m")

    checkpoint_callback = ModelCheckpoint(
        dirpath=hp_config["checkpoint_dir"],
        filename="cp-{epoch:1d}-{step:02d}",
        every_n_train_steps=hp_config["save_interval"],
        save_top_k=-1  # keep all checkpoints!
    )
 
    strategy = FSDPStrategy(
        auto_wrap_policy={Block},
        activation_checkpointing_policy={Block},
        state_dict_type="sharded",
        limit_all_gathers=True, # should make training faster when gpu memory is utilized almost fully
        # forward_prefetch=True,  # Enable forward prefetch for overlapping forward passes
        # sharding_strategy="HYBRID_SHARD", # TODO: try on 2 nodes with (2, 8)?
        # device_mesh=(2, 8),
        cpu_offload=False,
    )

    current_time = datetime.now().strftime("%d.%m._%H:%M")
    wandb_logger = WandbLogger(project="learn-lightning", name="Gemma2 at " + current_time)
    wandb_logger.watch(model, log="parameters", log_graph=False)

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

    trainer = L.Trainer(
        max_epochs=hp_config["max_epochs"],
        num_nodes=2,
        devices=8,
        strategy=strategy,
        precision=precision,
        logger=wandb_logger,
        callbacks=[checkpoint_callback,
                    # progress_bar
                    ],
        # max_steps=max_iters,
        # limit_train_batches=11,
        # limit_val_batches=eval_iters,
        accumulate_grad_batches=hp_config["accumulate_batch_size"] // hp_config["micro_batch_size"],
        log_every_n_steps=hp_config["log_interval"],
        val_check_interval=hp_config["val_interval"],
        # deterministic=True,
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
    )

    print("Starting training")
    trainer.fit(model, data_module) # , ckpt_path=hp_config["checkpoint_path"])
    if trainer.strategy.root_device.type == "cuda":
        trainer.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Save a simple checkpoint:
    trainer.save_checkpoint(hp_config["checkpoint_dir"] + "/" + "gemma2_finished.ckpt")

    trainer.print(torch.cuda.memory_summary())

    torch.save(model.state_dict(), 'pth_models/model_gemma2_trained.pth')


if __name__ == "__main__":
    hp_config_path = "config.yaml"
    main(hp_config_path)
