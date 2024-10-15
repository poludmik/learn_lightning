from datetime import datetime
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.callbacks.progress import RichProgressBar
from lightning.pytorch.plugins.environments import SLURMEnvironment
import signal
from BigDataModule import MyDataModule
from GemmaModule import Gemma2Finetuner

L.seed_everything(228, workers=True)

# warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

# Parameters
data_file = "../dataset-playground/azure_data/czech_llm_data/czech-llm-dataset-complete/cswiki_only.bin"

max_epochs = 1

# Initialize the DataModule
data_module = MyDataModule(
    data_file=data_file,
    block_size=2048,
    batch_size=1,
    num_workers=1,
)
data_module.setup()

# Calculate total steps
num_training_batches = len(data_module.train_dataloader())
print(f"Number of training batches: {num_training_batches}")
total_steps = max_epochs * num_training_batches
warmup_steps = int(0.001 * total_steps)  # 1% of total steps
print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

# Initialize the trained model
model = Gemma2Finetuner(warmup_steps=warmup_steps, total_steps=total_steps)

checkpoint_callback = ModelCheckpoint(
        dirpath="my_gemma2_cswiki_checkpoints",
        filename="cp-{epoch:1d}-{step:02d}",
        every_n_train_steps=2000,
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

layers = {
        nn.TransformerEncoderLayer,
        nn.TransformerDecoderLayer,
    }

strategy = FSDPStrategy(
    auto_wrap_policy=layers,
    activation_checkpointing_policy=layers,
    cpu_offload=False,
    state_dict_type="sharded",
    # limit_all_gathers=True, # for cuda crashing?
)

# Initialize the WandbLogger
current_time = datetime.now().strftime("%d.%m._%H:%M")
wandb_logger = WandbLogger(project="learn-lightning", name="Gemma2 at " + current_time)
wandb_logger.watch(model, log="parameters", log_graph=False)


# Initialize the Trainer
trainer = L.Trainer(
    max_epochs=max_epochs,
    logger=wandb_logger,
    callbacks=[checkpoint_callback, 
            #    progress_bar # doesn't work in slurm output files
            ],
    num_nodes=2,
    devices=8,
    # strategy='fsdp', # TODO: try simple "fsdp"
    strategy="deepspeed_stage_3", 
    precision="bf16-mixed",
    val_check_interval=100, # every N steps, check validation. Or set to 0.25 to check every 25% of 1 epoch
    # limit_train_batches=0.004,
    # limit_val_batches=0.1,
    # overfit_batches=0.001,
    deterministic=True,
    plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)]
)

#Start training
trainer.fit(
    model,
    datamodule=data_module,
    # ckpt_path="my_gemma2_cswiki_checkpoints/cp-epoch=0-step=2000.ckpt",
    # ckpt_path="my_gpt2_big_checkpoints/cp-epoch=0-step=100.ckpt", # if resuming with 1 epoch limit, it will start from the number of batches it has already seen and stop when it sees everything. that is, now it will take total_num_batches - already_seen_batches steps and end.
)
trainer.print(torch.cuda.memory_summary())

torch.save(model.state_dict(), 'pth_models/model_gemma2_cswiki.pth')
