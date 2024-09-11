"""basically the Jupyter notebook from Nvidia Deep Learning course"""
from typing import cast

import torch
import torch.cuda
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import Tensor

# Visualization tools
import torchvision.transforms.v2 as transforms
from torchvision.datasets import MNIST


def get_batch_accuracy(output: Tensor, y: Tensor, N: int) -> float:
    pred: Tensor = output.argmax(dim=1, keepdim=True)
    correct: int = pred.eq(y.view_as(pred)).sum().item()
    return correct / N


def train(device: torch.device, model: nn.Sequential, optimizer: Adam, loss_function: callable, train_loader: DataLoader) -> None:
    """train the model"""
    loss: int = 0
    accuracy: int = 0
    train_N: int = len(train_loader.dataset)  # type: ignore # https://github.com/pytorch/pytorch/issues/47055

    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)  # map data to the device's requirements
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))


def validate(device: torch.device, model: nn.Sequential, loss_function: callable, valid_loader: DataLoader) -> None:
    """validate the trained model"""
    loss: int = 0
    accuracy: int = 0
    validation_N: int = len(valid_loader.dataset)  # type: ignore # https://github.com/pytorch/pytorch/issues/47055

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)

            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, validation_N)
    print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))


def deep_learning() -> None:
    """nvidia deep learning, 1st example"""
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")

    train_set: MNIST = MNIST("./data/", train=True, download=True)
    valid_set: MNIST = MNIST("./data/", train=False, download=True)

    # original logic is deprecated
    # trans: transforms.Compose = transforms.Compose([transforms.ToTensor()])
    # https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html
    # basically transforms a PIL image to data 0->1.0.
    #
    trans: transforms.Compose = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
    train_set.transform = trans
    valid_set.transform = trans

    batch_size: int = 32

    train_loader: DataLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader: DataLoader = DataLoader(valid_set, batch_size=batch_size)

    # flattening image
    input_size: int = 1 * 28 * 28
    n_classes: int = 10
    layers = [
        nn.Flatten(),
        nn.Linear(input_size, 512),  # Input
        nn.ReLU(),  # Activation for input
        nn.Linear(512, 512),  # Hidden
        nn.ReLU(),  # Activation for hidden
        nn.Linear(512, n_classes)  # Output
    ]

    print(f"{layers=}")

    model: nn.Sequential = nn.Sequential(*layers)
    model.to(device)
    compiled_model = torch.compile(model)

    # loss function https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    loss_function = nn.CrossEntropyLoss()

    # optimizer https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    optimizer = Adam(compiled_model.parameters())  # type: ignore  # hidden type

    epochs: int = 5
    for epoch in range(epochs):
        print('Epoch: {}'.format(epoch))
        train(device, model, optimizer, loss_function, train_loader)
        validate(device, model, loss_function, valid_loader)


if __name__ == '__main__':
    deep_learning()
