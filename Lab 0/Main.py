from Task_0_1_LeakyReLU import CNN

# Import CIFAR-10 dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# Import torch
import torch

# Import transfroms
from torchvision import transforms

from TensorBoardHandler import TensorBoardHandler


def Task_0_1():
    # This is the wrapper for our tensorboard.
    tensorBoard = TensorBoardHandler()

    # Define the hyperparameters
    num_epochs = 20
    batch_size = 64
    learning_rate = 1e-4

    # Define the number of epochs
    # Define the number of epochs
    epochs = [epoch for epoch in range(num_epochs)]

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
    best_accuracy_SGD, best_epoch_SGD, SGD_train_losses, SGD_val_losses, SGD_val_accuracies = model.train_model(
        train_loader, validation_loader, loss_fn, optimizer, num_epochs)

    # Tensorboard logging
    tensorBoard.log_scalar("loss/train", SGD_train_losses, epochs)
    tensorBoard.log_scalar("loss/val", SGD_val_losses, epochs)
    tensorBoard.log_scalar("accuracy/val", SGD_val_accuracies, epochs)

    # Report on the Test accuracy and prints the models best result on training.
    test_accuracy_SGD = model.test_model(test_loader)
    print(
        f"Task 0.1 - Using SGD optimizer",
        f"Best validation accuracy: {best_accuracy_SGD} at epoch {best_epoch_SGD} - Best model test set accuracy: {test_accuracy_SGD}")

    # Create the model
    ADAM_model = CNN()

    # Defines the optimizer
    optimizer = torch.optim.Adam(ADAM_model.parameters(), lr=learning_rate)

    # Train the model
    best_accuracy_ADAM, best_epoch_ADAM, train_losses_ADAM, val_losses_ADAM, val_accuracies_ADAM = ADAM_model.train_model(
        train_loader, validation_loader, loss_fn, optimizer, num_epochs)

    # Tensorboard logging
    tensorBoard.log_scalar("loss/train", train_losses_ADAM, epochs)
    tensorBoard.log_scalar("loss/val", val_losses_ADAM, epochs)
    tensorBoard.log_scalar("accuracy/val", val_accuracies_ADAM, epochs)

    # Report on the Test accuracy and prints the models best result on training.
    test_accuracy_ADAM = ADAM_model.test_model(test_loader)
    print(
        f"Task 0.1 - Using ADAM optimizer",
        f"Best validation accuracy: {best_accuracy_ADAM} at epoch {best_epoch_ADAM} - Best model test set accuracy: {test_accuracy_ADAM}")

    import Task_0_1_Tanh

    SGD_model_Tanh = Task_0_1_Tanh.CNN()
    ADAM_model_Tanh = Task_0_1_Tanh.CNN()

    # Defines the optimizer
    SGD_optimizer = torch.optim.SGD(SGD_model_Tanh.parameters(), lr=learning_rate)
    ADAM_optimizer = torch.optim.Adam(ADAM_model_Tanh.parameters(), lr=learning_rate)

    # Train the model
    best_accuracy_SGD_Tanh, best_epoch_SGD_Tanh, train_losses_SGD_Tanh, val_losses_SGD_Tanh, val_accuracies_SGD_Tanh = SGD_model_Tanh.train_model(
        train_loader, validation_loader, loss_fn, SGD_optimizer, num_epochs)
    best_accuracy_ADAM_Tanh, best_epoch_ADAM_Tanh, train_losses_ADAM_Tanh, val_losses_ADAM_Tanh, val_accuracies_ADAM_Tanh = ADAM_model_Tanh.train_model(
        train_loader, validation_loader, loss_fn, ADAM_optimizer, num_epochs)

    # Tensorboard logging for SGD tanh
    tensorBoard.log_scalar("loss/train", train_losses_SGD_Tanh, epochs)
    tensorBoard.log_scalar("loss/val", val_losses_SGD_Tanh, epochs)
    tensorBoard.log_scalar("accuracy/val", val_accuracies_SGD_Tanh, epochs)

    # Tensorboard logging for ADAM tanh
    tensorBoard.log_scalar("loss/train", train_losses_ADAM_Tanh, epochs)
    tensorBoard.log_scalar("loss/val", val_losses_ADAM_Tanh, epochs)
    tensorBoard.log_scalar("accuracy/val", val_accuracies_ADAM_Tanh, epochs)

    # Report on the Test accuracy and prints the models best result on training.
    test_accuracy_SGD_Tanh = SGD_model_Tanh.test_model(test_loader)
    test_accuracy_ADAM_Tanh = ADAM_model_Tanh.test_model(test_loader)
    print(
        f"Task 0.1 - Using Tanh activation function",
        f"Best validation accuracy using SGD: {best_accuracy_SGD_Tanh} at epoch {best_epoch_SGD_Tanh} - Best model test set accuracy: {test_accuracy_SGD_Tanh}",
        f"Best validation accuracy using ADAM: {best_accuracy_ADAM_Tanh} at epoch {best_epoch_ADAM_Tanh} - Best model test set accuracy: {test_accuracy_ADAM_Tanh}",
        sep="\n")


def Task_0_2():
    from Task_0_2 import Task_0_2_1, Task_0_2_2

    # Task 0.2.1
    Task_0_2_1()

    # Task 0.2.2
    Task_0_2_2()


def main():
    # This is the first task of the lab 0
    #Task_0_1()

    # This is the second task of the lab 0
    Task_0_2()


if __name__ == "__main__":
    main()
