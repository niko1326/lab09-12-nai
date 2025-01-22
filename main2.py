import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import numpy as np

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
train_set = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_set = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)


# Define a simple model
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleCNN().to(device)

# Define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Train the model
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")


train_model(model, train_loader, criterion, optimizer)


# Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return y_true, y_pred


y_true, y_pred = evaluate_model(model, test_loader)

# Task 1: Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=train_set.classes,
            yticklabels=train_set.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Task 2: Precision, Recall, F1
report = classification_report(y_true, y_pred, target_names=train_set.classes)
print("Classification Report:")
print(report)

# Task 3: Visualize and transform
sample_images, sample_labels = iter(test_loader).next()
plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(sample_images[i][0].cpu().numpy(), cmap="gray")
    plt.title(f"Label: {train_set.classes[sample_labels[i]]}")
    plt.axis("off")
plt.show()

# Homework: Class distribution
train_classes = [label for _, label in train_set]
test_classes = [label for _, label in test_set]

plt.figure(figsize=(10, 5))
plt.hist(train_classes, bins=np.arange(11) - 0.5, alpha=0.7, label="Train")
plt.hist(test_classes, bins=np.arange(11) - 0.5, alpha=0.7, label="Test")
plt.xticks(range(10), train_set.classes, rotation=45)
plt.legend()
plt.title("Class Distribution in Train and Test Sets")
plt.xlabel("Classes")
plt.ylabel("Number of Samples")
plt.show()
