import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model with one convolutional layer
class NeuralNetworkV1(nn.Module):
    def __init__(self):
        super(NeuralNetworkV1, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # One convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 14 * 14, 128),  # Smaller fully connected layers
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        logits = self.fc_layers(x)
        return logits

# Model with two convolutional layers
class NeuralNetworkV2(nn.Module):
    def __init__(self):
        super(NeuralNetworkV2, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # First convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Second convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),  # Smaller fully connected layers
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        logits = self.fc_layers(x)
        return logits

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Train and save models
def train_and_save_model(model, model_name, epochs=5):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Training {model_name}")
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
    print(f"Done training {model_name}!")

    torch.save(model.state_dict(), f"{model_name}.pth")
    print(f"Model saved to {model_name}.pth")

# Train both models
train_and_save_model(NeuralNetworkV1(), "model_v1_Adam", epochs=5)
train_and_save_model(NeuralNetworkV2(), "model_v2_Adam", epochs=5)
