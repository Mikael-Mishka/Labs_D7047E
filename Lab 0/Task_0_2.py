import torch

# Imports ToTensor
from torchvision.transforms import ToTensor

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



def Task_0_2_2():
    print()
