import torch
import time

from torch import nn
from torchvision.datasets import MNIST





#Deta spliting


#traning_data
#valedasun_data
#test_data

#CNN

class Flatten(nn.Module):
    def forword(self, x):
        return x.view(x.size(0), -1)


class CNN_2(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # If cuda is available, use it
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        conulusen_layer = nn.Sequential(
            nn.Conv2d(3, 32, 5, (1, 1), padding=1, device=self.device),
            nn.Tanh(),
            nn.MaxPool2d(5, 1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5, (1, 1), padding=1, device=self.device),
            nn.Tanh(),
            nn.MaxPool2d(5, 1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, (1, 1), padding=1, device=self.device),
            nn.Tanh(),
            nn.MaxPool2d(5, 1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, (1, 1), padding=1, device=self.device),
            nn.Tanh(),
            nn.MaxPool2d(5, 1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, 3, (1, 1), padding=1, device=self.device),
            nn.Tanh(),
            nn.MaxPool2d(5, 1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 3, 3, (1, 1), padding= "same", device=self.device),
            nn.Tanh(),
        )

        conulusen_layer.to(self.device)

        # Dummy pass to get the ouput tensor size of the conv layers
        dummy = torch.randn(1, 3, 32, 32).to(self.device)

        # Get the output size of the conv layers
        conv_out = conulusen_layer(dummy)

        num_features = Flatten().forward(conv_out).shape[1]

        self.internal_model = nn.Sequential(
            conulusen_layer,
            Flatten(),
            nn.Linear(num_features, 200),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, 10),
        )
    def Forword(self, x):
        return self.intarnal_model(x)
    #the CNN model


    #model tranig
        #trening step


        #valedesun step


#adversarial images
class adversarial_images():

    #random imits


    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import numpy as np



    # Set a random seed for reproducibility
    torch.manual_seed(0)

    # Load the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize for MNIST
    ])

    # Download and load the MNIST dataset
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Split the dataset into training and test sets
    train_set, val_set = torch.utils.data.random_split(mnist_dataset, [20000, 4000])

    # DataLoaders for training and testing
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(val_set, batch_size=64, shuffle=False)


    # Define a simple CNN architecture for MNIST
    class Flatten(nn.Module):
        def forword(self, x):
            return x.view(x.size(0), -1)

    class CNN_2(nn.Module):
        kernel_1 = 5
        kernel_2 = 3
        samme_1 = kernel_1 // 2
        samme_2 = kernel_2 // 2
        def __init__(self, kernel_1=None, kernel_2=None, samme_1=None, samme_2=None, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # If cuda is available, use it
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

            conulusen_layer = nn.Sequential(
                nn.Conv2d(1, 32, kernel_1, (1, 1), padding=samme_1, device=self.device),
                nn.Tanh(),
                nn.MaxPool2d(5, 1),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, kernel_1, (1, 1), padding=samme_1, device=self.device),
                nn.Tanh(),
                nn.MaxPool2d(5, 1),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_2, (1, 1), padding=samme_2, device=self.device),
                nn.Tanh(),
                nn.MaxPool2d(5, 1),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 32, kernel_2, (1, 1), padding=samme_2, device=self.device),
                nn.Tanh(),
                nn.MaxPool2d(5, 1),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 16, kernel_2, (1, 1), padding=samme_2, device=self.device),
                nn.Tanh(),
                nn.MaxPool2d(5, 1),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, 3, kernel_2, (1, 1), padding=samme_2, device=self.device),
                nn.Tanh(),
            )

            conulusen_layer.to(self.device)

            # Dummy pass to get the ouput tensor size of the conv layers
            dummy = torch.randn(1, 1, 28, 28).to(self.device)

            # Get the output size of the conv layers
            conv_out = conulusen_layer(dummy)

            num_features = Flatten().forward(conv_out).shape[1]

            self.internal_model = nn.Sequential(
                conulusen_layer,
                Flatten(),
                nn.Linear(num_features, 200),
                nn.Tanh(),
                nn.Dropout(0.2),
                nn.Linear(200, 200),
                nn.Tanh(),
                nn.Dropout(0.2),
                nn.Linear(200, 10),
            )

        def Forword(self, x):
            return self.intarnal_model(x)


    # Create and train the model
    model = CNN_2()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(5):  # Train for 5 epochs
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Test the model to ensure it's working properly
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    print("Model Accuracy: {:.2f}%".format(accuracy * 100))

    # Get the indices of '4's in the test set
    index_of_4s = [i for i, t in enumerate(train_set.targets) if t == 4]

    # Choose a random '4' image
    import random
    rand_index = random.randint(0, len(index_of_4s) - 1)
    image_4 = train_set.data[index_of_4s[rand_index]].float() / 255.0  # normalize to [0, 1]

    # Define the target label as '9' (one-hot encoded)
    label_9 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float32).unsqueeze(0)

    # Set the image to require gradients
    image_4 = image_4.unsqueeze(0).unsqueeze(0)  # add batch and channel dimensions
    image_4.requires_grad = True

    # Perform forward pass and calculate loss
    output = model(image_4)
    loss = criterion(output, torch.argmax(label_9, dim=1))  # Target label '9'

    # Backpropagation to compute gradients
    model.zero_grad()
    loss.backward()

    # Define FGSM attack function
    def fgsm_attack(image, epsilon, gradient):
        sign_gradient = gradient.sign()
        perturbed_image = image + epsilon * sign_gradient
        perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Ensure values stay within [0, 1]
        return perturbed_image

    # Generate adversarial perturbation with epsilon
    epsilon = 0.1  # Strength of perturbation
    gradients = image_4.grad.data
    adversarial_image = fgsm_attack(image_4, epsilon, gradients)

    # Check if the adversarial image is classified as '9'
    with torch.no_grad():
        perturbed_output = model(adversarial_image)
        _, predicted = torch.max(perturbed_output.data, 1)

    print(f"Original Label: 4, Predicted Label after Adversarial Attack: {predicted.item()}")




    # Create random noise
    random_noise = torch.rand((1, 1, 28, 28))  # 28x28 single-channel random noise image

    # Check if the random noise image is classified as '9'
    with torch.no_grad():
        noise_output = model(random_noise)
        _, noise_predicted = torch.max(noise_output.data, 1)

    print(f"Random Noise Predicted Label: {noise_predicted.item()}")

















































import random
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import numpy as np

# Set a random seed for reproducibility
torch.manual_seed(0)

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # Normalize for MNIST
])

# Download and load the MNIST dataset
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split the dataset into training and test sets
train_set, val_set = torch.utils.data.random_split(mnist_dataset, [20000, 4000])

# DataLoaders for training and testing
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)


# Define a simple CNN architecture for MNIST
class Flatten(nn.Module):
    def forword(self, x):
        return x.view(x.size(0), -1)

class CNN_2(nn.Module):
    kernel_1 = 5
    kernel_2 = 3
    samme_1 = kernel_1 // 2
    samme_2 = kernel_2 // 2

    def __init__(self, kernel_1=None, kernel_2=None, samme_1=None, samme_2=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # If cuda is available, use it
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        conulusen_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_1, (1, 1), padding=samme_1, device=self.device),
            nn.Tanh(),
            nn.MaxPool2d(5, 1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_1, (1, 1), padding=samme_1, device=self.device),
            nn.Tanh(),
            nn.MaxPool2d(5, 1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_2, (1, 1), padding=samme_2, device=self.device),
            nn.Tanh(),
            nn.MaxPool2d(5, 1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_2, (1, 1), padding=samme_2, device=self.device),
            nn.Tanh(),
            nn.MaxPool2d(5, 1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, kernel_2, (1, 1), padding=samme_2, device=self.device),
            nn.Tanh(),
            nn.MaxPool2d(5, 1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 3, kernel_2, (1, 1), padding=samme_2, device=self.device),
            nn.Tanh(),
        )

        conulusen_layer.to(self.device)

        # Dummy pass to get the ouput tensor size of the conv layers
        dummy = torch.randn(1, 1, 28, 28).to(self.device)

        # Get the output size of the conv layers
        conv_out = conulusen_layer(dummy)

        num_features = Flatten().forward(conv_out).shape[1]

        self.internal_model = nn.Sequential(
            conulusen_layer,
            Flatten(),
            nn.Linear(num_features, 200),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(200, 10),
        )

    def Forword(self, x):
        return self.intarnal_model(x)


    # CNNModel, Loss Function (cross-entropy loss), and optimizer (Adam optimizer)
    model = CNN_2()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        train_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}')

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        val_acc = correct / len(val_loader.dataset)
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_acc:.4f}')
        model.train()





    # Get the indices of '4's in the test set
    index_of_4s = [i for i, t in enumerate(train_set.targets) if t == 4]

    # Choose a random '4' image
    rand_index = random.randint(0, len(index_of_4s) - 1)
    image_4 = train_set.data[index_of_4s[rand_index]].float() / 255.0  # normalize to [0, 1]

    # Define the target label as '9' (one-hot encoded)
    label_9 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float32).unsqueeze(0)

    # Set the image to require gradients
    image_4 = image_4.unsqueeze(0).unsqueeze(0)  # add batch and channel dimensions
    image_4.requires_grad = True

    # Perform forward pass and calculate loss
    output = model(image_4)
    loss = criterion(output, torch.argmax(label_9, dim=1))  # Target label '9'

    # Backpropagation to compute gradients
    model.zero_grad()
    loss.backward()


    # Define FGSM attack function
    def fgsm_attack(image, epsilon, gradient):
        sign_gradient = gradient.sign()
        perturbed_image = image + epsilon * sign_gradient
        perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Ensure values stay within [0, 1]
        return perturbed_image


    # Generate adversarial perturbation with epsilon
    epsilon = 0.1  # Strength of perturbation
    gradients = image_4.grad.data
    adversarial_image = fgsm_attack(image_4, epsilon, gradients)

    # Check if the adversarial image is classified as '9'
    with torch.no_grad():
        perturbed_output = model(adversarial_image)
        _, predicted = torch.max(perturbed_output.data, 1)

    print(f"Original Label: 4, Predicted Label after Adversarial Attack: {predicted.item()}")

    # Create random noise
    random_noise = torch.rand((1, 1, 28, 28))  # 28x28 single-channel random noise image

    # Check if the random noise image is classified as '9'
    with torch.no_grad():
        noise_output = model(random_noise)
        _, noise_predicted = torch.max(noise_output.data, 1)

    print(f"Random Noise Predicted Label: {noise_predicted.item()}")