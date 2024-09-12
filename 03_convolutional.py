"""Lesson #3, Convolutional Neural Networks"""
from typing import Any

import torch.nn as nn
import pandas as pd
import torch
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

target_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# target_device: torch.device = torch.device("cpu")
print(f"{torch.cuda.is_available()=}")
print(f"{target_device=}")

# 3.2.1 Preparing image
train_df = pd.read_csv("data/asl_data/sign_mnist_train.csv")
valid_df = pd.read_csv("data/asl_data/sign_mnist_valid.csv")

sample_df = train_df.head().copy()  # Grab the top 5 rows
sample_df.pop('label')
sample_x = sample_df.values
print(f"{sample_x=}")
print(f"{sample_x.shape=}")

IMG_HEIGHT: int = 28
IMG_WIDTH: int = 28
IMG_CHS: int = 1

sample_x = sample_x.reshape(-1, IMG_CHS, IMG_HEIGHT, IMG_WIDTH)
print(f"{sample_x.shape=}")


# 3.2.2 Create a Dataset
class MyDataset(Dataset):
    def __init__(self, base_df, target_device: torch.device):
        x_df = base_df.copy()  # Some operations below are in-place
        y_df = x_df.pop('label')
        x_df = x_df.values / 255  # Normalize values from 0 to 1
        x_df = x_df.reshape(-1, IMG_CHS, IMG_WIDTH, IMG_HEIGHT)
        self.xs: Tensor = torch.tensor(x_df).float().to(target_device)
        self.ys: Tensor = torch.tensor(y_df).to(target_device)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y

    def __len__(self):
        return len(self.xs)


# 3.2.3 Create a DataLoader
BATCH_SIZE: int = 32

train_data = MyDataset(train_df, target_device)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
train_N: int = len(train_loader.dataset)

valid_data = MyDataset(valid_df, target_device)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
valid_N: int = len(valid_loader.dataset)

batch = next(iter(train_loader))
print(f"{batch=}")
print(f"{batch[0].shape=}")
print(f"{batch[1].shape=}")
print("\n")


# 3.3 Creating a Convolutional Model
n_classes: int = 24
kernel_size: int = 3
flattened_img_size: int = 75 * 3 * 3

original_model: nn.Sequential = nn.Sequential(
    # First convolution
    nn.Conv2d(IMG_CHS, 25, kernel_size, stride=1, padding=1),  # 25 x 28 x 28
    nn.BatchNorm2d(25),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),  # 25 x 14 x 14
    # Second convolution
    nn.Conv2d(25, 50, kernel_size, stride=1, padding=1),  # 50 x 14 x 14
    nn.BatchNorm2d(50),
    nn.ReLU(),
    nn.Dropout(.2),
    nn.MaxPool2d(2, stride=2),  # 50 x 7 x 7
    # Third convolution
    nn.Conv2d(50, 75, kernel_size, stride=1, padding=1),  # 75 x 7 x 7
    nn.BatchNorm2d(75),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),  # 75 x 3 x 3
    # Flatten to Dense
    nn.Flatten(),
    nn.Linear(flattened_img_size, 512),
    nn.Dropout(.3),
    nn.ReLU(),
    nn.Linear(512, n_classes)
)


# 3.4 Summarizing the Model
compiled_model = torch.compile(original_model.to(target_device))
print(f"{compiled_model=}")

loss_function = nn.CrossEntropyLoss()
optimizer = Adam(compiled_model.parameters())


def get_batch_accuracy(output: Tensor, y: Tensor, size: int):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / size


# 3.5 Training the Model
def train(device: torch.device, model: Any, optimizer: Adam,
          loss_function: callable, train_loader: DataLoader) -> None:
    """train the model"""
    loss: int = 0
    accuracy: int = 0
    size: int = len(train_loader.dataset)  # type: ignore # https://github.com/pytorch/pytorch/issues/47055

    model.train()
    for x, y in train_loader:
        output: Tensor = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, size)

    print(f'Train - {loss=:.4f} {accuracy=:.4f}')


def validate(device: torch.device, model: Any, loss_function: callable, valid_loader: DataLoader) -> None:
    """validate the trained model"""
    loss: int = 0
    accuracy: int = 0
    size: int = len(valid_loader.dataset)  # type: ignore # https://github.com/pytorch/pytorch/issues/47055

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            output = model(x)

            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, size)
    print(f'Valid - {loss=:.4f} {accuracy=:.4f}')


epochs: int = 20
for epoch in range(epochs):
    print(f'Epoch: {epoch}')
    train(target_device, compiled_model, optimizer, loss_function, train_loader)
    validate(target_device, compiled_model, loss_function, valid_loader)
