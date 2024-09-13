import os
from datetime import datetime

import torch
import wandb
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data
import lightning as L
from lightning.pytorch.callbacks import ModelSummary, DeviceStatsMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything

seed_everything(228)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index], 0

    def __len__(self):
        return len(self.data)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder, lr=0.5e-3, *args, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
        self.save_hyperparameters() # save the hyperparameters to the checkpoint object and to wandb (logs arguments to the __init__ method)
        self.total_loss = 0

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)

        # Accumulate loss and count steps
        self.total_loss += loss.item()

        # Log average loss every 100 steps
        if self.global_step % 100 == 0:
            avg_loss = self.total_loss / 100
            self.log("train_loss_avg", avg_loss, prog_bar=True)
            self.total_loss = 0  # Reset after logging

        self.log("train_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        # print(f'val_loss: {val_loss}')
        self.log("val_loss", val_loss, prog_bar=True) # automatically averages across the validation step
        return val_loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


checkpoint_callback = ModelCheckpoint(
    dirpath="my_callback_checkpoints",
    filename="sample-mnist-{epoch:1d}-{step:02d}",
    every_n_train_steps=30000,
    save_top_k=-1  # keep all checkpoints!
)

current_time = datetime.now().strftime("%d.%m._%H:%M")
wandb_logger = WandbLogger(project="learn-lightning", name="autoencoder at " + current_time)

dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
test_set = MNIST(root="MNIST", download=True, train=False, transform=transforms.ToTensor())

# use 20% of training data for validation
train_set_size = int(len(dataset) * 0.8)
valid_set_size = len(dataset) - train_set_size

# split the train set into two
train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size])

train_loader = DataLoader(dataset, num_workers=11)
# print(next(iter(train_loader)))

valid_loader = DataLoader(valid_set, num_workers=11)


# model
autoencoder = LitAutoEncoder(Encoder(), Decoder())
# autoencoder.load_state_dict(torch.load('model.pth'))
wandb_logger.watch(autoencoder, log="parameters")

# train model
trainer = L.Trainer(
    max_epochs=5,
    callbacks=[checkpoint_callback, ModelSummary(max_depth=-1), DeviceStatsMonitor()], # to ensure that you’re using the full capacity of your accelerator (GPU/TPU/HPU).
    # fast_dev_run=5, # run only 5 batches to check if the whole code is working (train, val, test). will disable tuner, checkpoint callbacks, early stopping callbacks, loggers
    limit_train_batches=0.05, # run only 5% of the training data
    limit_val_batches=0.05, # run only 10% of the validation data
    num_sanity_val_steps=2, # run 2 steps of validation before training to check it is working
    # profiler="simple",
    # log_every_n_steps=10, # chatgpt generated
    # logger=True,
    val_check_interval=500, # every 500 steps, check validation. Or set to 0.25 to check every 25% of 1 epoch
    logger=wandb_logger,
)
trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=valid_loader)
trainer.test(autoencoder, dataloaders=DataLoader(test_set, num_workers=11))

# save model
torch.save(autoencoder.state_dict(), 'model.pth')

# load model
autoencoder.load_state_dict(torch.load('model.pth', weights_only=True))

# predict
autoencoder.eval()
x = torch.randn(1, 28, 28)
x = x.view(x.size(0), -1)
z = autoencoder.encoder(x)
# print(z.shape) # torch.Size([1, 3])
# z = torch.randn(1, 3)
x_hat = autoencoder.decoder(z)
# print(x_hat.shape) # torch.Size([1, 784])

# show the image
import matplotlib.pyplot as plt
x_hat = x_hat.view(1, 28, 28)
x_hat = x_hat.detach().numpy()
# plt.imshow(x_hat[0])
# plt.show()

