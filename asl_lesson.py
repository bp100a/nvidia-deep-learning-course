import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

train_df = pd.read_csv("data/asl_data/sign_mnist_train.csv")
valid_df = pd.read_csv("data/asl_data/sign_mnist_valid.csv")

print(f"{train_df.head()=}")

# 2.3.3 Extracting the labels
y_train = train_df.pop('label')
y_valid = valid_df.pop('label')
print(f"{y_train=}")

# 2.3.4 Extracting the images
x_train = train_df.values
x_valid = valid_df.values
print(f"{x_train=}")

# 2.3.5 Summarizing the training

print(f"{x_train.shape=}, {y_train.shape=}")
print(f"{x_valid.shape=}, {y_valid.shape=}")


# 2.4 Visualizing the Data
import matplotlib.pyplot as plt
plt.figure(figsize=(40,40))

num_images: int = 20
for i in range(num_images):
    row = x_train[i]
    label = y_train[i]

    image = row.reshape(28,28)
    plt.subplot(1, num_images, i+1)
    plt.title(label, fontdict={'fontsize': 30})
    plt.axis('off')
    plt.imshow(image, cmap='gray')

plt.show()

max_value: int = x_train.max()
x_train = train_df.values / max_value
x_valid = valid_df.values / max_value


# 2.4.2 Custom Datasets

class MyDataset(Dataset):
    def __init__(self, x_df, y_df):
        self.xs = torch.tensor(x_df).float().to(device)
        self.ys = torch.tensor(y_df).to(device)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y

    def __len__(self):
        return len(self.xs)


BATCH_SIZE: int = 32

train_data = MyDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
train_N = len(train_loader.dataset)

valid_data = MyDataset(x_valid, y_valid)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
valid_N = len(valid_loader.dataset)


# Build the Model
input_size:int  = 28 * 28
n_classes: int = 26

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(input_size, 512),  # Input
    nn.ReLU(),  # Activation for input
    nn.Linear(512, 512),  # Hidden
    nn.ReLU(),  # Activation for hidden
    nn.Linear(512, n_classes)  # Output
)

model = torch.compile(model.to(device))
print(f"{model=}")

loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

# 2.6 Training the model


def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N


def train():
    loss = 0
    accuracy = 0

    model.train()
    for x, y in train_loader:
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    print(f'Train - {loss=:.4f} {accuracy=:.4f}')


def validate():
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            output = model(x)

            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    print(f'Valid - {loss=:.4f} {accuracy=:.4f}')

# 2.6.3 The training loop

epochs: int = 20
for epoch in range(epochs):
    print(f'{epoch=}')
    train()
    validate()
