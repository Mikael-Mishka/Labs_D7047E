from Task_0_1 import CNN

# Import CIFAR-10 dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# Import torch
import torch

# Import transfroms
from torchvision import transforms

def Task_0_1():
    # Define the hyperparameters
    num_epochs = 10
    batch_size = 64
    learning_rate = 1e-4

    # Create the model
    model = CNN()

    # Load the data
    train_data = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
    test_validation_data = CIFAR10(root='./data', train=False, download=True, transform=ToTensor())

    # Create the loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Here we add training augmentation
    train_loader.dataset.transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        ])


    # Calculates the amount of samples in the test and validation set each
    test_size = len(test_validation_data) // 2
    validation_size = len(test_validation_data) - test_size

    # Split the test and validation set
    test_data, validation_data = torch.utils.data.random_split(test_validation_data, [test_size, validation_size])

    # Create the test and validation loaders
    test_loader = DataLoader(test_data, batch_size=batch_size)
    validation_loader = DataLoader(validation_data, batch_size=batch_size)

    # Defines the optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train the model
    best_accuracy_SGD, best_epoch_SGD = model.train_model(train_loader, validation_loader, loss_fn, optimizer, num_epochs)

    # Report on the Test accuracy and prints the models best result on training.
    test_accuracy_SGD = model.test_model(test_loader)
    print(
        f"Task 0.1 - Using SGD optimizer",
        f"Best validation accuracy: {best_accuracy_SGD} at epoch {best_epoch_SGD} - Best model test set accuracy: {test_accuracy_SGD}")

    SGD_model = model

    # Create the model
    ADAM_model = CNN()

    # Defines the optimizer
    optimizer = torch.optim.Adam(ADAM_model.parameters(), lr=learning_rate)

    # Train the model
    best_accuracy_ADAM, best_epoch_ADAM = ADAM_model.train_model(train_loader, validation_loader, loss_fn, optimizer, num_epochs)

    # Report on the Test accuracy and prints the models best result on training.
    test_accuracy_ADAM = ADAM_model.test_model(test_loader)
    print(
        f"Task 0.1 - Using ADAM optimizer",
        f"Best validation accuracy: {best_accuracy_ADAM} at epoch {best_epoch_ADAM} - Best model test set accuracy: {test_accuracy_ADAM}")

def main():
    # This is the first task of the lab 0
    Task_0_1()


if __name__ == "__main__":
    main()