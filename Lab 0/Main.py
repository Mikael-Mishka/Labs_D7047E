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
    num_epochs = 10
    batch_size = 64
    learning_rate = 1e-4

    # Define the number of epochs
    # Define the number of epochs
    epochs = [epoch for epoch in range(num_epochs)]

    # Create the model
    model = CNN()

    print("""
-----------------------------------------------------------
0.1 Task
The following steps will guide you in creating your own experiment.
    • Download and prepare CIFAR-10 dataset (it is already available in the above mentioned libraries)
        – PyTorch
        – Keras and Tensorflow
-----------------------------------------------------------
            """)

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

    print("""
-----------------------------------------------------------
0.1 Task
    • Write a simple CNN network for classifying images
        – use LeakyReLU as the activation function (optional)
        – use SGD as the optimizer and 0.0001 as the learning rate, and keep all default parameters (optional)
        – Report the accuracy on the test set (optional)
    • Change the optimiser to Adam and run again the experiment. Report accuracy on test set.
-----------------------------------------------------------
                """)

    print("""
SOLUTION:     
    """)

    print(f"\n----- Task 0.1 (TRAINING)- Training SGD optimizer and LeakyReLU activation function -----\n")

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
        f"\n----- Task 0.1 (RESULTS)- Using SGD optimizer and LeakyReLU activation function -----",
        f"Best validation accuracy: {best_accuracy_SGD} at epoch {best_epoch_SGD} - Best model test set accuracy: {test_accuracy_SGD}\n", sep="\n")

    # Create the model
    ADAM_model = CNN()

    # Defines the optimizer
    optimizer = torch.optim.Adam(ADAM_model.parameters(), lr=learning_rate)

    print(f"\n----- Task 0.1 (TRAINING)- Training ADAM optimizer and LeakyReLU activation function -----\n")

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
        f"\n----- Task 0.1 (RESULTS)- Using ADAM optimizer and LeakyReLU activation function -----",
        f"Best validation accuracy: {max(val_accuracies_ADAM)} at epoch {best_epoch_ADAM} - Best model test set accuracy: {test_accuracy_ADAM}\n", sep="\n")

    print("""
-----------------------------------------------------------
0.1 Task
    • Swap the LeakyReLUs for Tanh. Then run again the experiment and report accuracy on
      test set. Make a separate file for this experiment.
-----------------------------------------------------------
    """)

    import Task_0_1_Tanh

    SGD_model_Tanh = Task_0_1_Tanh.CNN()
    ADAM_model_Tanh = Task_0_1_Tanh.CNN()

    # Defines the optimizer
    SGD_optimizer = torch.optim.SGD(SGD_model_Tanh.parameters(), lr=learning_rate)
    ADAM_optimizer = torch.optim.Adam(ADAM_model_Tanh.parameters(), lr=learning_rate)

    print(f"\n----- Task 0.1 (TRAINING)- Training SGD optimizer and Tanh activation function -----\n")

    # Train the models
    best_accuracy_SGD_Tanh, best_epoch_SGD_Tanh, train_losses_SGD_Tanh, val_losses_SGD_Tanh, val_accuracies_SGD_Tanh = SGD_model_Tanh.train_model(
        train_loader, validation_loader, loss_fn, SGD_optimizer, num_epochs)

    # Test SGD model
    test_accuracy_SGD_Tanh = SGD_model_Tanh.test_model(test_loader)
    print(
        f"\n----- Task 0.1 (RESULTS)- Using SGD optimizer and Tanh activation function -----",
        f"Best validation accuracy using SGD: {best_accuracy_SGD_Tanh} at epoch {best_epoch_SGD_Tanh} - Best model test set accuracy: {test_accuracy_SGD_Tanh}",
        f"",sep="\n")

    # Tensorboard logging for SGD tanh
    tensorBoard.log_scalar("loss/train", train_losses_SGD_Tanh, epochs)
    tensorBoard.log_scalar("loss/val", val_losses_SGD_Tanh, epochs)
    tensorBoard.log_scalar("accuracy/val", val_accuracies_SGD_Tanh, epochs)

    print(f"\n----- Task 0.1 (TRAINING)- Training ADAM optimizer and Tanh activation function -----")

    best_accuracy_ADAM_Tanh, best_epoch_ADAM_Tanh, train_losses_ADAM_Tanh, val_losses_ADAM_Tanh, val_accuracies_ADAM_Tanh = ADAM_model_Tanh.train_model(
        train_loader, validation_loader, loss_fn, ADAM_optimizer, num_epochs)

    # Tensorboard logging for ADAM tanh
    tensorBoard.log_scalar("loss/train", train_losses_ADAM_Tanh, epochs)
    tensorBoard.log_scalar("loss/val", val_losses_ADAM_Tanh, epochs)
    tensorBoard.log_scalar("accuracy/val", val_accuracies_ADAM_Tanh, epochs)

    # Test ADAM model
    test_accuracy_ADAM_Tanh = ADAM_model_Tanh.test_model(test_loader)


    print(
        f"\n----- Task 0.1 (RESULTS)- Using ADAM optimizer and Tanh activation function -----",
        f"Best validation accuracy using ADAM: {best_accuracy_ADAM_Tanh} at epoch {best_epoch_ADAM_Tanh} - Best model test set accuracy: {test_accuracy_ADAM_Tanh}",
        sep="\n")

    print("""
-----------------------------------------------------------
Task 0.1 (DONE)
-----------------------------------------------------------

""")


def Task_0_2():
    from Task_0_2 import Task_0_2_1, Task_0_2_2

    # Task 0.2.1
    Task_0_2_1()

    print("""
-----------------------------------------------------------
0.2.1 Transfer Learning from ImageNet (DONE)
-----------------------------------------------------------
    """)

    # Task 0.2.2
    Task_0_2_2()


def main():
    # This is the first task of the lab 0
    Task_0_1()

    # This is the second task of the lab 0
    Task_0_2()


if __name__ == "__main__":
    main()
