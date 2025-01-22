import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import sys
import time
import matplotlib.pyplot as plt

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define the models
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
            nn.Linear(16 * 14 * 14, 128),  # Fully connected layers
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        logits = self.fc_layers(x)
        return logits

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
            nn.Linear(32 * 7 * 7, 64),  # Fully connected layers
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        logits = self.fc_layers(x)
        return logits

# Load the model file
if len(sys.argv) != 3:
    print("Usage: python detect.py <model_version> <image_path>")
    sys.exit(1)

model_version = sys.argv[1]
image_path = sys.argv[2]

# Load the appropriate model
if model_version == "v1_SGD":
    model = NeuralNetworkV1()
    model_file = "model_v1_SGD.pth"
elif model_version == "v2_SGD":
    model = NeuralNetworkV2()
    model_file = "model_v2_SGD.pth"
elif model_version == "v1_Adam":
    model = NeuralNetworkV1()
    model_file = "model_v1_Adam.pth"
elif model_version == "v2_Adam":
    model = NeuralNetworkV2()
    model_file = "model_v2_Adam.pth"
elif model_version == "v1_Adam_AUG":
    model = NeuralNetworkV1()
    model_file = "model_v1_Adam_AUG.pth"
elif model_version == "v2_Adam_AUG":
    model = NeuralNetworkV2()
    model_file = "model_v2_Adam_AUG.pth"
elif model_version == "v2_Adam_AUG_new":
    model = NeuralNetworkV2()
    model_file = "model_v2_Adam_AUG_new.pth"
else:
    print("Invalid model version. Choose 'v1_SGD', 'v2_SGD', 'v1_Adam', 'v2_Adam'.")
    sys.exit(1)

# Load the model weights
start_time = time.time()
model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
model.to(device)
model.eval()
print(f"Model {model_version} loaded in {time.time() - start_time:.2f} seconds")

# Image transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the image
start_time = time.time()
image = Image.open(image_path)
print(f"Image loaded in {time.time() - start_time:.2f} seconds")

# Transform the image
start_time = time.time()
image_tensor = transform(image).unsqueeze(0).to(device)
print(f"Image transformed in {time.time() - start_time:.2f} seconds")

# Make a prediction
start_time = time.time()
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_class_idx = probabilities.argmax(dim=1, keepdim=True).item()
print(f"Prediction made in {time.time() - start_time:.2f} seconds")

# Map of classes (FashionMNIST)
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Get the predicted label and probability
predicted_label = labels[predicted_class_idx]
predicted_probability = probabilities[0, predicted_class_idx].item()

# Display the input image with the prediction
plt.imshow(image, cmap="gray")
plt.title(f"Prediction: {predicted_label} ({predicted_probability:.2f})")
plt.axis("off")
plt.show()

# Display the class probabilities as a bar chart
plt.bar(labels, probabilities[0].cpu().numpy())
plt.title("Class Probabilities")
plt.xlabel("Class")
plt.ylabel("Probability")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
