import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"


class NeuralNetworkV2(nn.Module):
    def __init__(self):
        super(NeuralNetworkV2, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        logits = self.fc_layers(x)
        return logits

# Ustawienia danych z augmentacją
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomInvert(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Zbiór treningowy
train_dataset = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform_train)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Zbiór testowy
test_dataset = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform_test)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Funkcja kosztu i optymalizator
loss_fn = nn.CrossEntropyLoss()

# Funkcja treningu
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Obliczanie predykcji i straty
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Funkcja do trenowania i zapisywania modelu
def train_and_save(model_class, model_name):
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Zmiana na Adam
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
    print("Done!")
    torch.save(model.state_dict(), f"{model_name}.pth")
    print(f"Model saved to {model_name}.pth")

print("Training NeuralNetworkV2 with Augmentation")
train_and_save(NeuralNetworkV2, "model_v2_with_aug")
