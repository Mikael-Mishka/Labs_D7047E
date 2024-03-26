from Task_0_1 import CNN

# Import CIFAR-10 dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# Import torch
import torch

def Task_0_1():
    # Define the hyperparameters
    num_epochs = 40
    batch_size = 64
    learning_rate = 1e-3

    # Create the model
    model = CNN()

    # Load the data
    train_data = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
    test_validation_data = CIFAR10(root='./data', train=False, download=True, transform=ToTensor())

    # Create the loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Calculates the amount of samples in the test and validation set each
    test_size = len(test_validation_data) // 2
    validation_size = len(test_validation_data) - test_size

    # Split the test and validation set
    test_data, validation_data = torch.utils.data.random_split(test_validation_data, [test_size, validation_size])

    # Create the test and validation loaders
    test_loader = DataLoader(test_data, batch_size=batch_size)
    validation_loader = DataLoader(validation_data, batch_size=batch_size)

    # Defines the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train the model
    model.train_model(train_loader, validation_loader, loss_fn, optimizer, num_epochs)


def main():
    # This is the first task of the lab 0
    Task_0_1()


if __name__ == "__main__":
    main()