import numpy as np
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from sklearn import preprocessing
from enum import Enum
import copy
# import SGD for optimizer
from torch.optim import SGD

# import Adam for optimizer
from torch.optim import Adam

# to measure the performance import L1Loss
from torch.nn import L1Loss
from torch.utils.data import random_split


training_y = np.load('/Users/huzuwang/我的雲端硬碟/code/python/projects/ADL/ADL_(IM5062)/111ntu-homework2/training_y.npy')


training_x_grid = np.load('/Users/huzuwang/我的雲端硬碟/code/python/projects/ADL/ADL_(IM5062)/111ntu-homework2/training_x_grid.npy')


class CustomStarDataset(Dataset):
    # This loads the data and converts it, make data rdy
    def __init__(self):
        # load data
        self.df = training_x_grid
        # extract labels
        self.df_labels = training_y
        # conver to torch dtypes
        self.dataset = torch.tensor(self.df).float()

        self.labels = torch.tensor(self.df_labels).long()

    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.dataset)

    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]


ds = CustomStarDataset()



# 8760, 7665(105*73), 1095(15*73)
train_ds, val_ds = random_split(ds, [7665, 1095])
len(train_ds), len(val_ds)

from torch.utils.data import DataLoader

batch_size = 73

train_loader = DataLoader(train_ds,batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

testing_x_grid = np.load('/Users/huzuwang/我的雲端硬碟/code/python/projects/ADL/ADL_(IM5062)/111ntu-homework2/testing_x_grid.npy')
testing_x_grid_T = torch.tensor(testing_x_grid.astype(np.float32))

from torch.nn import L1Loss


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss_fc1 = L1Loss()
        loss = loss_fc1(out, labels)
        # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss_fc2 = L1Loss()
        loss2 = loss_fc2(out, labels)
        # Calculate loss
        return {'val_loss': loss2.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss']))


class Cifar10CnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # input 30, 38, 13
            nn.Conv2d(30, 32, kernel_size=2, padding=1),
            # input 30 channel(RGB) and output 32
            # output: 32  38, 13
            nn.ReLU(),  # doesn't change shape
            # output: 32  38, 13
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # output: 64  38, 13
            nn.ReLU(),
            # output: 64  38, 13
            nn.MaxPool2d(2, 2),  # output: 64 x 19 x 7

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 9 x 3

            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(),
            nn.Linear(3456, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2))

        # nn.Flatten(),
        # nn.Linear(8512,512),
        # nn.ReLU(),
        # nn.Linear(512,256),
        # nn.ReLU(),
        # nn.Linear(256, 64),
        # nn.ReLU(),
        # nn.Linear(64, 2))

    def forward(self, xb):
        return self.network(xb)

@torch.no_grad()  # means that will not do any gradients while evaluate
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    #print("1")
    for epoch in range(epochs):
        # Training Phase
        #print("2")
        model.train()
        #print("3")
        train_losses = []
        for batch in train_loader:
            #print("4")
            loss = model.training_step(batch)
            #print("5")
            train_losses.append(loss)
            #print("6")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


model = Cifar10CnnModel()
num_epochs = 10
opt_func = torch.optim.Adam
lr = 0.005

history = fit(num_epochs, lr, model, train_loader, val_loader, opt_func)

num_epochs = 5
opt_func = torch.optim.Adam
lr = 0.001

history = fit(num_epochs, lr, model, train_loader, val_loader, opt_func)
FILE = 'hw2_7_1.pt'
torch.save(model, FILE)
out = model(testing_x_grid_T)

