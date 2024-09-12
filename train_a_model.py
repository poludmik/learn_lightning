import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data
import lightning as L


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
    def __init__(self, encoder, decoder):
        super().__init__()
        self.save_hyperparameters() # save the hyperparameters to the checkpoint object
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        # print(f'loss: {loss}')
        # self.log("global_step", self.global_step)
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
        self.log("val_loss", val_loss)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.5e-3)
        return optimizer


checkpoint_callback = ModelCheckpoint(
    dirpath="my_callback_checkpoints",
    # monitor="global_step",
    filename="sample-mnist-{epoch:02d}-{step:02d}",
    every_n_train_steps=200,
    save_top_k=-1  # keep all checkpoints!
)


dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
test_set = MNIST(root="MNIST", download=True, train=False, transform=transforms.ToTensor())

# use 20% of training data for validation
train_set_size = int(len(dataset) * 0.8)
valid_set_size = len(dataset) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size], generator=seed)

train_loader = DataLoader(dataset)
print(next(iter(train_loader)))

valid_loader = DataLoader(valid_set)


# model
autoencoder = LitAutoEncoder(Encoder(), Decoder())
autoencoder.load_state_dict(torch.load('model.pth'))

# train model
trainer = L.Trainer(
    max_epochs=1,
    callbacks=[checkpoint_callback]
)
trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=valid_loader)
trainer.test(autoencoder, dataloaders=DataLoader(test_set))

# save model
torch.save(autoencoder.state_dict(), 'model.pth')

# load model
autoencoder.load_state_dict(torch.load('model.pth'))

# predict
autoencoder.eval()
x = torch.randn(1, 28, 28)
x = x.view(x.size(0), -1)
z = autoencoder.encoder(x)
print(z.shape) # torch.Size([1, 3])
# z = torch.randn(1, 3)
x_hat = autoencoder.decoder(z)
print(x_hat.shape) # torch.Size([1, 784])

# show the image
import matplotlib.pyplot as plt
x_hat = x_hat.view(1, 28, 28)
x_hat = x_hat.detach().numpy()
plt.imshow(x_hat[0])
plt.show()
