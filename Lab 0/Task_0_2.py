import copy

import torch

# Imports nn
import torch.nn as nn

# Imports ToTensor
from torchvision.transforms import ToTensor

from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10

# Imports random split
from torch.utils.data import random_split, DataLoader

# Import transforms
import torchvision.transforms as transforms


def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total * 100


def train_model(model, train_loader, validation_loader, loss_fn, optimizer, num_epochs,
                device):
    import time

    # Here is the metrics we will be tracking
    best_accuracy = 0
    best_epoch = 0

    model.to(device)
    print(f"Got into training loop")

    validation_accuracies = []
    train_losses = []
    validation_losses = []

    # Here is the main training loop
    for epoch in range(num_epochs):

        # Reset the metrics
        current_accuracy = 0
        current_train_loss = 0
        current_validation_loss = 0

        # Sets us in training mode
        model.train()

        # Training loop
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Track the loss
            current_train_loss += loss.item()

        # Sets us in evaluation mode
        model.eval()

        # Validation loop
        with torch.no_grad():
            for i, (images, labels) in enumerate(validation_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                # Track the loss
                current_validation_loss += loss.item()

                # Track the accuracy
                _, predicted = torch.max(outputs.data, 1)
                current_accuracy += (predicted == labels).sum().item()

        # Calculate the accuracy
        current_accuracy = current_accuracy / len(validation_loader.dataset) * 100
        validation_accuracies.append(current_accuracy)

        # Calculate the average loss
        current_train_loss /= len(train_loader)
        current_validation_loss /= len(validation_loader)

        # Appends the losses
        train_losses.append(current_train_loss)
        validation_losses.append(current_validation_loss)

        # Print the results
        print(f"Epoch: {epoch} - Train loss: {current_train_loss} - Validation loss: {current_validation_loss} - Validation accuracy: {current_accuracy}")



def Task_0_2_1():
    # 0.2.1 Transfer Learning from ImageNet

    # Hyperparameters
    batch_size = 64
    learning_rate = 1e-4
    num_epochs = 1

    # Define transforms to resize images to 224x224 (AlexNet input size)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Download and prepare the CIFAR-10 dataset
    train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_and_test_data = CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Split the validation and test set
    test_size = len(val_and_test_data) // 2
    val_size = len(val_and_test_data) - test_size
    val_data, test_data = random_split(val_and_test_data, [val_size, test_size])

    # Create the loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Gets cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the model (AlexNet)
    model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    model.to(device)

    # Change the last layer to fit CIFAR-10
    model.classifier[6] = torch.nn.Linear(4096, 10)

    train_model(model, train_loader, val_loader, torch.nn.CrossEntropyLoss(),
                torch.optim.Adam(model.parameters(), lr=learning_rate), num_epochs, device)

    test_accuracy = test_model(model, test_loader)

    print(f"AlexNet trained on CIFAR-10 - Test accuracy: {test_accuracy} %")

    # Alexnet feauture extraction

    # define the model
    fx_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    fx_model.to(device)

    # Freeze the models parameters in all layers and change the last layer to fit CIFAR-10
    for param in fx_model.parameters():
        param.requires_grad = False

    fx_model.classifier[6] = torch.nn.Linear(4096, 10)

    # Train last layer
    train_model(fx_model, train_loader, val_loader, torch.nn.CrossEntropyLoss(),
                torch.optim.Adam(fx_model.parameters(), lr=learning_rate), num_epochs, device)

    test_accuracy = test_model(fx_model, test_loader)

    print(f"AlexNet feature extraction on CIFAR-10 - Test accuracy: {test_accuracy} %")

def train_model(self, train_loader, validation_loader, loss_fn, optimizer, num_epochs, device):

    # Here is the metrics we will be tracking
    best_accuracy = 0
    best_epoch = 0

    self.to(device)

    validation_accuracies = []
    train_losses = []
    validation_losses = []

    # Here is the main training loop
    for epoch in range(num_epochs):
        # Reset the metrics
        current_accuracy = 0
        current_train_loss = 0
        current_validation_loss = 0

        # Sets us in training mode
        self.train()

        # Training loop
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = self(images)
            loss = loss_fn(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Track the loss
            current_train_loss += loss.item()

        # Sets us in evaluation mode
        self.eval()

        # Validation loop
        with torch.no_grad():
            for i, (images, labels) in enumerate(validation_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = self(images)
                loss = loss_fn(outputs, labels)

                # Track the loss
                current_validation_loss += loss.item()

                # Track the accuracy
                _, predicted = torch.max(outputs.data, 1)
                current_accuracy += (predicted == labels).sum().item()

        # Calculate the accuracy
        current_accuracy = current_accuracy / len(validation_loader.dataset) * 100
        validation_accuracies.append(current_accuracy)

        # Calculate the average loss
        current_train_loss /= len(train_loader)
        current_validation_loss /= len(validation_loader)

        # Appends the losses
        train_losses.append(current_train_loss)
        validation_losses.append(current_validation_loss)

        # Print the results
        print(f"Epoch: {epoch} - Train loss: {current_train_loss} - Validation loss: {current_validation_loss} - Validation accuracy: {current_accuracy}")

    return best_accuracy, best_epoch, train_losses, validation_losses, validation_accuracies


from torch import nn
import torch
import time


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNN(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # If cuda is available, use it
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 5, (1, 1), padding=1, device=self.device),
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
            nn.Conv2d(16, 1, 3, (1, 1), padding=1, device=self.device),
            nn.Tanh(),
        )

        self.conv_layers.to(self.device)

        # Dummy pass to get the ouput tensor size of the conv layers
        dummy = torch.randn(1, 1, 28, 28).to(self.device)

        # Get the output size of the conv layers
        conv_out = self.conv_layers(dummy)

        num_features = Flatten().forward(conv_out).shape[1]

        self.internal_model = nn.Sequential(
            self.conv_layers,
            Flatten(),
            nn.Linear(num_features, 200),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(200, 10),
        )

        self.internal_model.to(self.device)

    def forward(self, x):
        return self.internal_model(x)

    def train_model(self, train_loader, validation_loader, loss_fn, optimizer, num_epochs):

        self.to(self.device)

        # Here is the metrics we will be tracking
        best_accuracy = 0
        best_epoch = 0

        last_print = time.time()

        validation_accuracies = []
        validation_losses = []
        train_losses = []

        # Here is the main training loop
        for epoch in range(num_epochs):

            # Reset the metrics
            current_accuracy = 0
            current_train_loss = 0
            current_validation_loss = 0

            # Sets us in training mode
            self.train()

            # Training loop
            for i, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()

                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self(images)
                loss = loss_fn(outputs, labels)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                # Track the loss
                current_train_loss += loss.item()

            # Sets us in evaluation mode
            self.eval()

            # Validation loop
            with torch.no_grad():
                for i, (images, labels) in enumerate(validation_loader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass
                    outputs = self(images)
                    loss = loss_fn(outputs, labels)

                    # Track the loss
                    current_validation_loss += loss.item()

                    # Track the accuracy
                    _, predicted = torch.max(outputs, 1)
                    correct = (predicted == labels).sum().item()
                    total = labels.size(0)
                    current_accuracy += (correct / total)

            # Calculate the average loss and accuracy
            current_train_loss /= len(train_loader)
            current_validation_loss /= len(validation_loader)
            current_accuracy = current_accuracy/len(validation_loader)*100

            # Append the results
            train_losses.append(current_train_loss)
            validation_losses.append(current_validation_loss)
            validation_accuracies.append(current_accuracy)

            if time.time() - last_print > 15:
                last_print = time.time()
                # Print the results
                print(
                    f"Epoch: {epoch}/{num_epochs}, Train Loss: {current_train_loss}, Validation Loss: {current_validation_loss}, Accuracy: {current_accuracy} %")

            # Check if we have a new best accuracy
            if current_accuracy > best_accuracy:
                torch.save(self, "best_model.pth")
                best_accuracy = current_accuracy
                best_epoch = epoch
        print(f"Best epoch: {best_epoch}\tBest accuracy: {best_accuracy}")
        return best_accuracy, best_epoch, train_losses, validation_losses, validation_accuracies


    def test_model(self, test_loader):
        best_model = torch.load("best_model.pth")
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = best_model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total*100

def Task_0_2_2():
    # Transfer learning from MNIST


    # Hyperparameters
    batch_size = 64
    learning_rate = 1e-4
    num_epochs = 10

    # Prepare the MNIST dataset
    transform = ToTensor()
    train_data = MNIST(root='./data', train=True, download=True, transform=transform)
    val_and_test_data = MNIST(root='./data', train=False, download=True, transform=transform)

    # Calculates the test and validation size
    test_size = len(val_and_test_data) // 2
    val_size = len(val_and_test_data) - test_size

    # Split the validation and test set
    val_data, test_data = random_split(val_and_test_data, [val_size, test_size])

    # Creates the loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Gets cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the model
    transfer_model = CNN()

    # We use our own CNN model
    _, _, train_losses, validation_losses, validation_accuracies = train_model(transfer_model, train_loader, val_loader, torch.nn.CrossEntropyLoss(), torch.optim.Adam(transfer_model.parameters(), lr=learning_rate), num_epochs, device)

    print(f"Training on MNIST gives us the best accuracy of: {max(validation_accuracies)}",
          f"Lowest train loss: {min(train_losses)}",
          f"Lowest validation loss: {min(validation_losses)}",
          sep="\n")

    # Use the transfer model as a pretrained model on SVHN

    # Import SVHN
    from torchvision.datasets import SVHN

    # Prepare SVHN dataset
    transform = ToTensor()
    train_data = SVHN(root='./data', split='train', download=True, transform=transform)
    val_and_test_data = SVHN(root='./data', split='test', download=True, transform=transform)

    # Calculates the test and validation size
    test_size = len(val_and_test_data) // 2
    val_size = len(val_and_test_data) - test_size

    # Split the validation and test set
    val_data, test_data = random_split(val_and_test_data, [val_size, test_size])

    # Creates the loader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Test the model on SVHN
    test_accuracy = transfer_model.test_model(test_loader)

    print(f"Transfer model trained on MNIST tested on SVHN - Test accuracy: {test_accuracy} %")

    # Changes the first convolution layer such that it accepts 3 channels instead of 1
    transfer_model.conv_layers[0] = nn.Conv2d(3, 32, 5, (1, 1), padding=1).to(device)

    # Changes the transfer_models dummy pass to get the first linear layer size,
    dummy_data = torch.randn(1, 3, 32, 32).to(device)
    conv_out = transfer_model.conv_layers(dummy_data)
    num_features = Flatten().forward(conv_out).shape[1]

    # Changes the first linear layer to fit SVHN
    transfer_model.internal_model[2] = nn.Linear(num_features, 200)

    # Train the model on SVHN
    _, _, train_losses, validation_losses, validation_accuracies = transfer_model.train_model(train_loader, val_loader, torch.nn.CrossEntropyLoss(), torch.optim.Adam(transfer_model.parameters(), lr=learning_rate), num_epochs)

    # Test the model on SVHN
    test_accuracy = transfer_model.test_model(test_loader)

    print(f"Transfer model trained on SVHN - Test accuracy: {test_accuracy} %")



