"""American Sign Language (ASL) learning example"""
# 4. Data augmentation
import os
from typing import Any

import torch.nn as nn
import pandas as pd
import torch
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
from pandas.core.interchange.dataframe_protocol import DataFrame
import matplotlib.pyplot as plt

import utils


class MyDataset(Dataset):
    def __init__(self, base_df):
        x_df = base_df.copy()
        y_df = x_df.pop('label')
        x_df = x_df.values / 255  # Normalize values from 0 to 1
        x_df = x_df.reshape(-1, IMG_CHS, IMG_WIDTH, IMG_HEIGHT)
        self.xs = torch.tensor(x_df).float().to(device)
        self.ys = torch.tensor(y_df).to(device)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y

    def __len__(self):
        return len(self.xs)


class MyConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p):
        kernel_size = 3
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.MaxPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.model(x)


def get_batch_accuracy(output: Tensor, y: Tensor, size: int) -> float:
    """determine accuracy of the batch"""
    pred: Tensor = output.argmax(dim=1, keepdim=True)
    correct: int = pred.eq(y.view_as(pred)).sum().item()
    return correct / size


def train(model: Any, random_transforms: transforms.Compose, loss_function: callable) -> None:
    loss: int = 0
    accuracy: int = 0
    model.train()
    for x, y in train_loader:
        output = model(random_transforms(x))  # Updated
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()
        # compute the loss/accuracy for this session
        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    print(f'Train - {loss=:.4f} {accuracy=:.4f}')


def validate(model: Any, loss_function: callable) -> None:
    loss: int = 0
    accuracy: int = 0

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            output = model(x)
            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    print(f'Valid - {loss=:.4f} {accuracy=:.4f}')


if __name__ == "__main__":
    """run the script"""

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"{torch.cuda.is_available()=}, {device=}")
        major, minor = torch.cuda.get_device_capability()
        if major < 7:
            print(f"downgrade to CPU as CUDA Capability is {major}.{minor} ({torch.cuda.get_device_name()}), requires 7.0 or greater")
            device = torch.device("cpu")  # switch to CPU
            # torch._dynamo.config.suppress_errors = True

    IMG_HEIGHT: int = 28
    IMG_WIDTH: int = 28
    IMG_CHS: int = 1
    N_CLASSES: int = 24

    root_dir: str = os.path.dirname(__file__)
    train_df = pd.read_csv(os.path.join(root_dir, "data", "asl_data", "sign_mnist_train.csv"))
    valid_df = pd.read_csv(os.path.join(root_dir, "data", "asl_data","sign_mnist_valid.csv"))

    n: int = 32
    train_data = MyDataset(train_df)
    train_loader = DataLoader(train_data, batch_size=n, shuffle=True)
    train_N = len(train_loader.dataset)

    valid_data = MyDataset(valid_df)
    valid_loader = DataLoader(valid_data, batch_size=n)
    valid_N = len(valid_loader.dataset)

    # create the model that will process this using the Convolution block and the flattened inputs
    flattened_img_size = 75 * 3 * 3

    # Input 1 x 28 x 28
    base_model = nn.Sequential(
        MyConvBlock(IMG_CHS, 25, 0),  # 25 x 14 x 14
        MyConvBlock(25, 50, 0.2),  # 50 x 7 x 7
        MyConvBlock(50, 75, 0),  # 75 x 3 x 3
        # Flatten to Dense Layers
        nn.Flatten(),
        nn.Linear(flattened_img_size, 512),
        nn.Dropout(.3),
        nn.ReLU(),
        nn.Linear(512, N_CLASSES)
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(base_model.parameters())

    model = torch.compile(base_model.to(device))
    print(f"{model=}")

    # before training, adjust (augment) the data
    row_0: pd.DataFrame = train_df.head(1)
    y_0: pd.Series = row_0.pop('label')
    x0: pd.array = row_0.values / 255
    x0 = x0.reshape(IMG_CHS, IMG_WIDTH, IMG_HEIGHT)
    x_0: torch.Tensor = torch.tensor(x0)
    print(f"{x_0.shape=}")

    image = F.to_pil_image(x_0)
    plt.imshow(image, cmap='gray')
    plt.show()  # not in a Jupyter notebook so make it visible

    # random, horizontal flip
    trans = transforms.Compose([
        transforms.RandomHorizontalFlip()
    ])

    new_x_0 = trans(x_0)
    image = F.to_pil_image(new_x_0)
    plt.xlabel("horizontal flip")
    plt.imshow(image, cmap='gray')
    plt.show()

    # random rotation
    trans = transforms.Compose([
        transforms.RandomRotation(10)
    ])

    new_x_0 = trans(x_0)
    image = F.to_pil_image(new_x_0)
    plt.xlabel("rotation")
    plt.imshow(image, cmap='gray')
    plt.show()

    # jitter the brightness
    brightness = .2  # Change to be from 0 to 1
    contrast = .5  # Change to be from 0 to 1

    trans = transforms.Compose([
        transforms.ColorJitter(brightness=brightness, contrast=contrast)
    ])

    new_x_0 = trans(x_0)
    image = F.to_pil_image(new_x_0)
    plt.xlabel("jitter")
    plt.imshow(image, cmap='gray')
    plt.show()

    # Compose
    random_transforms = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), scale=(.9, 1), ratio=(1, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.2, contrast=.5)
    ])
    new_x_0 = random_transforms(x_0)
    image = F.to_pil_image(new_x_0)
    plt.xlabel("compose")
    plt.imshow(image, cmap='gray')
    plt.show()

    # time to train
    epochs: int = 20
    for epoch in range(epochs):
        print(f'Epoch: {epoch=}')
        train(model, random_transforms, loss_function)
        validate(model, loss_function)
