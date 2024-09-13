"""5b - Presidential Doggy Door"""
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms

import glob
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device=}")

from torchvision.models import vgg16
from torchvision.models import VGG16_Weights

# load the VGG16 network *pre-trained* on the ImageNet dataset
weights = VGG16_Weights.DEFAULT
vgg_model = vgg16(weights=weights)
print(f"{vgg_model.to(device)=}")

# Freeze the model
vgg_model.requires_grad_(False)
print("VGG16 Frozen")

# 5b.2.3 Add new layer to the model
N_CLASSES: int = 1

my_model = nn.Sequential(
    vgg_model,  # original, frozen model
    nn.Linear(1000, N_CLASSES)  # reduce the output classifier of the first model to 1 prediction, it's Bo!
)

print(f"{my_model.to(device)=}")

# Verify that only the latest model layer will learn
for idx, param in enumerate(my_model.parameters()):
    frozen: str = 'stubborn' if not param.requires_grad else 'open-minded'
    print(f"Layer{idx}, {frozen}")


# 5.2.4 Compiling the model
loss_function = nn.BCEWithLogitsLoss()
optimizer = Adam(my_model.parameters())
my_model = my_model.to(device)

# 5b.3 - Data Augmentation
pre_trans = weights.transforms()


class MyDataset(Dataset):
    DATA_LABELS: list[str] = ["bo", "not_bo"]

    def __init__(self, data_dir):
        self.imgs: list = []
        self.labels: list = []

        for l_idx, label in enumerate(self.DATA_LABELS):
            data_paths = glob.glob(data_dir + label + '/*.jpg', recursive=True)
            for path in data_paths:
                img = Image.open(path)
                self.imgs.append(pre_trans(img).to(device))
                self.labels.append(torch.tensor(l_idx).to(device).float())

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.imgs)

# 5b.3.2 - Data loaders
n: int = 32

train_path = "data/presidential_doggy_door/train/"
train_data = MyDataset(train_path)
train_loader = DataLoader(train_data, batch_size=n, shuffle=True)
train_N: int = len(train_loader.dataset)

valid_path = "data/presidential_doggy_door/valid/"
valid_data = MyDataset(valid_path)
valid_loader = DataLoader(valid_data, batch_size=n)
valid_N: int = len(valid_loader.dataset)

print(f"There are {train_N} training images and {valid_N} validation images")

# 5b.3.3 - more data
IMG_WIDTH, IMG_HEIGHT = (224, 224)

random_trans: transforms.Compose = transforms.Compose([
    transforms.RandomRotation(25),
    transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), scale=(.8, 1), ratio=(1, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2)
])

# 5b.4 - Training


def get_batch_accuracy(output, y, N: int) -> float:
    zero_tensor: Tensor = torch.tensor([0]).to(device)
    pred: Tensor = torch.gt(output, zero_tensor)
    correct: int = pred.eq(y.view_as(pred)).sum().item()
    return correct / N


def train(model, check_grad: bool = False) -> None:
    loss: float = 0.0
    accuracy: float = 0.0
    model.train()
    for x, y in train_loader:
        output: Tensor = torch.squeeze(model(random_trans(x)))
        optimizer.zero_grad()
        batch_loss: Tensor = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    if check_grad:
        print('Last Gradient:')
        for param in model.parameters():
            print(param.grad)
    print(f'Train - {loss=:.4f} {accuracy=:.4f}')


def validate(model) -> None:
    loss: float = 0.0
    accuracy: float = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            output: Tensor = torch.squeeze(model(x))
            loss += loss_function(output, y.float()).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    print(f'Valid - {loss=:.4f} {accuracy=:.4f}')

# Now learn something!
epochs: int = 10
for epoch in range(epochs):
    print(f'Epoch: {epoch}')
    train(my_model, check_grad=False)
    validate(my_model)


# expected output (from Nvidia Jupyter notebook)
# Epoch: 0
# Train - Loss: 0.6578 Accuracy: 0.9424
# Valid - Loss: 0.0832 Accuracy: 0.9667
# Epoch: 1
# Train - Loss: 0.9372 Accuracy: 0.9281
# Valid - Loss: 0.0881 Accuracy: 0.9667
# Epoch: 2
# Train - Loss: 0.7674 Accuracy: 0.9353
# Valid - Loss: 0.0906 Accuracy: 0.9667
# Epoch: 3
# Train - Loss: 0.9662 Accuracy: 0.9137
# Valid - Loss: 0.0981 Accuracy: 0.9667
# Epoch: 4
# Train - Loss: 1.0734 Accuracy: 0.9137
# Valid - Loss: 0.0924 Accuracy: 0.9667
# Epoch: 5
# Train - Loss: 0.6672 Accuracy: 0.9496
# Valid - Loss: 0.0846 Accuracy: 0.9667
# Epoch: 6
# Train - Loss: 1.3526 Accuracy: 0.9209
# Valid - Loss: 0.0784 Accuracy: 0.9667
# Epoch: 7
# Train - Loss: 0.5248 Accuracy: 0.9640
# Valid - Loss: 0.0559 Accuracy: 0.9667
# Epoch: 8
# Train - Loss: 1.0671 Accuracy: 0.8921
# Valid - Loss: 0.0518 Accuracy: 0.9667
# Epoch: 9
# Train - Loss: 0.6766 Accuracy: 0.9568
# Valid - Loss: 0.0632 Accuracy: 0.9667

# Fine Tuning the model
# Unfreeze the base model
vgg_model.requires_grad_(True)
optimizer = Adam(my_model.parameters(), lr=.000001)

epochs = 2
for epoch in range(epochs):
    print(f'Epoch: {epoch}')
    train(my_model, check_grad=False)
    validate(my_model)

# Examining the predictions

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image)
    plt.show()


def make_prediction(file_path) -> float:
    show_image(file_path)
    image = Image.open(file_path)
    image = pre_trans(image).to(device)
    image = image.unsqueeze(0)
    output = my_model(image)
    prediction: float = output.item()
    return prediction


print(f"{make_prediction('data/presidential_doggy_door/valid/bo/bo_20.jpg')=}")
print(f"{make_prediction('data/presidential_doggy_door/valid/not_bo/121.jpg')=}")


def presidential_doggy_door(image_path):
    pred: float = make_prediction(image_path)
    if pred < 0.0:
        print("It's Bo! Let him in!")
    else:
        print("That's not Bo! Stay out!")


presidential_doggy_door('data/presidential_doggy_door/valid/not_bo/131.jpg')
presidential_doggy_door('data/presidential_doggy_door/valid/bo/bo_29.jpg')
