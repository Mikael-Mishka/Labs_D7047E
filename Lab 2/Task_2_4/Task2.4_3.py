import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch import device
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Set a random seed for reproducibility
torch.manual_seed(0)

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor()
])

# Download and load the MNIST dataset
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


# DataLoaders for training and testing
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# Define a simple CNN architecture for MNIST
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CNN_2(nn.Module):
    kernel_1 = 5
    kernel_2 = 3
    samme_1 = kernel_1 // 2
    samme_2 = kernel_2 // 2

    def __init__(self):
        super().__init__()

        # If cuda is available, use it
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        convolution_layer = nn.Sequential(
            nn.Conv2d(1, 32, self.kernel_1, (1, 1), padding=self.samme_1, device=self.device),
            nn.Tanh(),
            nn.MaxPool2d(5, 1),
            nn.Conv2d(32, 64, self.kernel_1, (1, 1), padding=self.samme_1, device=self.device),
            nn.Tanh(),
            nn.MaxPool2d(5, 1),
            nn.Conv2d(64, 64, self.kernel_2, (1, 1), padding=self.samme_2, device=self.device),
            nn.Tanh(),
            nn.MaxPool2d(5, 1),
            nn.Conv2d(64, 32, self.kernel_2, (1, 1), padding=self.samme_2, device=self.device),
            nn.Tanh(),
            nn.MaxPool2d(5, 1),
            nn.Conv2d(32, 16, self.kernel_2, (1, 1), padding=self.samme_2, device=self.device),
            nn.Tanh(),
            nn.MaxPool2d(5, 1),
            nn.Conv2d(16, 3, self.kernel_2, (1, 1), padding=self.samme_2, device=self.device),
            nn.Tanh(),
        )

        convolution_layer.to(self.device)

        # Dummy pass to get the output tensor size of the conv layers
        dummy = torch.randn(1, 1, 28, 28).to(self.device)

        # Get the output size of the conv layers
        conv_out = convolution_layer(dummy)

        num_features = Flatten().forward(conv_out).shape[1]

        self.internal_model = nn.Sequential(
            convolution_layer,
            Flatten(),
            nn.Linear(num_features, 200),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(200, 10),
        )

    def forward(self, x):
        return self.internal_model(x)


# CNN Model, Loss Function (cross-entropy loss), and optimizer (Adam optimizer)
model = CNN_2()
model = model.to(model.device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training configuration
num_epochs = 1

# Training loop
for epoch in range(num_epochs):

    train_loss = 0.0
    val_loss = 0.0
    val_acc = 0.0

    model.train()

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(model.device), labels.to(model.device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(model.device), labels.to(model.device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            val_acc += correct / len(labels)

    val_acc /= len(val_loader)
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    val_acc = val_acc*100

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Validation loss: {val_loss} Validation Accuracy: {val_acc:.4f}')


# Adversarial Attack
index_of_4s = [i for i, (_, t) in enumerate(val_loader.dataset) if t == 4]

# Check if there are any '4' images
if not index_of_4s:
    raise ValueError("No images with label '4' found in the test set.")

# Randomly select a '4' image
rand_index = random.randint(0, len(index_of_4s) - 1)
image, _ = test_data[index_of_4s[rand_index]]
image_4 = image.float() / 255.0  # Normalize to [0, 1]


# Define the target label as '9' (one-hot encoded)
label_9 = torch.tensor([0,0,0,0,0,0,0,0,0,1])

print(label_9)

# Set the image to require gradients
image_4 = image_4.unsqueeze(0)  # Add batch and channel dimensions
image_4 = image_4.to(model.device)
image_4.requires_grad = True



# plot the image
import matplotlib.pyplot as plt
plt.imshow(image_4[0][0].cpu().detach().numpy(), cmap='gray')
plt.show()


image_4.requires_grad = True

# Perform forward pass and calculate loss
output = model(image_4)
print("OUTPUT SHOULD be corresponding to 4:", output)

label_9 = label_9.view(1, -1).to(model.device)
print(label_9.shape)

loss = criterion(output, torch.argmax(label_9, dim=1))  # Target label '9'
model.zero_grad()
loss.backward()


# Define FGSM attack function
def fgsm_attack(image, epsilon, gradient):
    sign_gradient = torch.sign(gradient)
    perturbed_image = image + epsilon * sign_gradient
    # Normalize the perturbed image
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# Generate adversarial perturbation with epsilon
epsilon = 0.00002  # Strength of perturbation

gradients = image_4.grad

adversarial_image = fgsm_attack(image_4, epsilon, gradients)

# Plot the adversarial image
plt.imshow(adversarial_image[0][0].cpu().detach().numpy(), cmap='gray')
plt.show()

# Check if the adversarial image is classified as '9'
with torch.no_grad():
    perturbed_output = model(adversarial_image)
    print("Perpurbed_output:",perturbed_output)
    predicted = torch.argmax(perturbed_output, 1)

print(f"Original Label: 4, Predicted Label after Adversarial Attack: {predicted.item()}")

# Create random noise
random_noise = torch.rand((1, 1, 28, 28)).to(device="cuda" if torch.cuda.is_available() else "cpu")  # 28x28 single-channel random noise image

# TODO: MORE TO BE DONE.