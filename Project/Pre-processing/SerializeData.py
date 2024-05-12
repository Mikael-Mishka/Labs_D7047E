# Imports path and file manipulation libraries
import os
import pathlib
from PIL import Image

import matplotlib.pyplot as plt

# data manipulation libraries and torchvision
import torch
import torchvision.transforms as transforms

import dill
import torch.nn as nn

from Project.Model import Custom_CNN


# Creates Labels and Data Tensors and loads them into memory by a return
def dataset_to_tensors(dataset_dir_path: pathlib.Path,
                       pickled_tensors_dir_path: pathlib.Path) -> tuple[list[torch.Tensor],
list[int],
list[torch.Tensor],
list[int],
list[torch.Tensor],
list[int]
]:
    """
    ASSUME:
        - DatasetResizedGrayscaled
            - train
                - Class1
                - Class2
            - test
                - Class1
                - Class2
            - val
                - Class1
                - Class2
    """

    # Create the lists to hold the tensors and labels
    train_data: list[torch.Tensor] = []
    train_labels: list[int] = []
    test_data: list[torch.Tensor] = []
    test_labels: list[int] = []
    val_data: list[torch.Tensor] = []
    val_labels: list[int] = []

    # ToTensor() transformation
    to_tensor = transforms.ToTensor()

    # Loop through the dataset directory
    for sub_directory in os.listdir(dataset_dir_path):
        sub_directory_path = dataset_dir_path / sub_directory

        # Loop through the sub-directory
        for class_directory in os.listdir(sub_directory_path):
            class_directory_path = sub_directory_path / class_directory

            # Loop through the class directory
            for image_path in os.listdir(class_directory_path):
                image_path = class_directory_path / image_path

                img = Image.open(image_path)

                tensor_data = to_tensor(img)

                # Append the tensor to the appropriate list
                # ASSUME 0 = NORMAL label and 1 = PNEUMONIA label
                if sub_directory == 'train':
                    train_data.append(tensor_data)
                    train_labels.append(0 if class_directory == 'NORMAL' else 1)

                elif sub_directory == 'test':
                    test_data.append(tensor_data)
                    test_labels.append(0 if class_directory == 'NORMAL' else 1)

                elif sub_directory == 'val':
                    val_data.append(tensor_data)
                    val_labels.append(0 if class_directory == 'NORMAL' else 1)

                else:
                    raise ValueError(f"Invalid sub-directory: {sub_directory}")

    return train_data, train_labels, test_data, test_labels, val_data, val_labels


def dill_tensors(train_data_tensor, train_labels, test_data_tensor, test_labels, val_data_tensor, val_labels):
    with open("train_data_tensor.dill", "wb") as f:
        dill.dump(train_data_tensor, f)

    with open("train_labels.dill", "wb") as f:
        dill.dump(train_labels, f)

    with open("test_data_tensor.dill", "wb") as f:
        dill.dump(test_data_tensor, f)

    with open("test_labels.dill", "wb") as f:
        dill.dump(test_labels, f)

    with open("val_data_tensor.dill", "wb") as f:
        dill.dump(val_data_tensor, f)

    with open("val_labels.dill", "wb") as f:
        dill.dump(val_labels, f)


def load_dilled_tensors() -> tuple[torch.Tensor,
                                   torch.Tensor,
                                   torch.Tensor,
                                   torch.Tensor,
                                   torch.Tensor,
                                   torch.Tensor]:

    with open("train_data_tensor.dill", "rb") as f:
        train_data_tensor = dill.load(f)

    with open("train_labels.dill", "rb") as f:
        train_labels = dill.load(f)

    with open("test_data_tensor.dill", "rb") as f:
        test_data_tensor = dill.load(f)

    with open("test_labels.dill", "rb") as f:
        test_labels = dill.load(f)

    with open("val_data_tensor.dill", "rb") as f:
        val_data_tensor = dill.load(f)

    with open("val_labels.dill", "rb") as f:
        val_labels = dill.load(f)

    return train_data_tensor, train_labels, test_data_tensor, test_labels, val_data_tensor, val_labels


def train_validate_model(model, train_loader, val_loader,
                criterion, optimizer, EPOCHS):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    best_val_loss = 999.0

    # Keep track of accuracies and losses
    train_losses = []
    val_losses = []
    val_accuracies = []

    print(f"TRAINING BEGINS NOW\n\n")
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomCrop(256, padding=10)
    ])

    for epoch in range(EPOCHS):

        train_loss = 0.0
        val_loss = 0.0

        val_accuracy = 0.0

        model.train()
        for i, (data, labels) in enumerate(train_loader):
            data = train_transforms(data)
            data = data.to(device)
            labels = labels.to(device).type(torch.long)

            optimizer.zero_grad()

            outputs = model(data)

            loss = criterion(outputs, labels.squeeze(1))

            loss.cpu().backward()

            optimizer.step()

            train_loss += loss.to('cpu').item()

        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        total = 0
        correct = 0

        with torch.no_grad():

            for i, (data, labels) in enumerate(val_loader):
                data = data.to(device)
                labels = labels.to(device).type(torch.long)

                outputs = model(data)

                loss = criterion(outputs, labels.squeeze(1))

                val_loss += loss.cpu().item()

                # Calculate the correctly classified images and the total
                total += labels.size(0)
                predicted = torch.max(outputs.data, 1)[1]

                correct += (predicted == labels.squeeze(1)).sum().item()

                val_accuracy += correct/total


        avg_val_accuracy = val_accuracy/len(val_loader)
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)

        print(f"Epoch: {epoch+1}/{EPOCHS}",
              f"Train Loss: {avg_train_loss}",
              f"Val Loss: {avg_val_loss}",
              f"Val Accuracy: {avg_val_accuracy}",
              sep="\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")


    return train_losses, val_losses, val_accuracies


def test_model(model, test_loader, criterion):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():

        test_loss = 0.0
        test_accuracy = 0.0

        for i, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            labels = labels.to(device).type(torch.long)

            outputs = model(data)

            loss = criterion(outputs, labels.squeeze(1))

            test_loss += loss.item()

            # Calculate validation accuracy
            total = labels.size(0)
            predicted = torch.max(outputs.data, 1)[1]
            correct = (predicted == labels.squeeze(1)).sum().item()

            test_accuracy += correct/total

        avg_test_accuracy = test_accuracy/len(test_loader)
        avg_test_loss = test_loss/len(test_loader)

        print(f"Test Loss: {avg_test_loss}",
              f"Test Accuracy: {avg_test_accuracy}",
              sep="\n")


def plot_train_val_test_metrics(train_losses,
                                val_losses,
                                val_accuracies,
                                epochs):

    # Create plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the losses
    ax[0].plot(range(epochs), train_losses, label="Train Loss")
    ax[0].plot(range(epochs), val_losses, label="Val Loss")
    ax[0].set_title("Losses")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    # Plot the accuracies
    ax[1].plot(range(epochs), val_accuracies, label="Val Accuracy")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    plt.show()




def run() -> None:

    # Set parent directory
    base_path = pathlib.Path(__file__).parent

    (train_data,
     train_labels,
     test_data,
     test_labels,
     val_data,
     val_labels) = dataset_to_tensors(
        dataset_dir_path=pathlib.Path(base_path / "DatasetResizedGrayscaled"),
        pickled_tensors_dir_path=pathlib.Path(base_path / "Picklies")
    )

    # print tests
    print(f"train_data: {len(train_data)}",
          f"train_labels: {len(train_labels)}",
          f"test_data: {len(test_data)}",
          f"test_labels: {len(test_labels)}",
          f"val_data: {len(val_data)}",
          f"val_labels: {len(val_labels)}",
          sep="\n")

    # Shapes
    print(f"train_data[0].shape: {train_data[0].shape}",
          f"train_labels[0]: {train_labels[0]}",
          f"test_data[0].shape: {test_data[0].shape}",
          f"test_labels[0]: {test_labels[0]}",
          f"val_data[0].shape: {val_data[0].shape}",
          f"val_labels[0]: {val_labels[0]}",
          sep="\n")

    # Convert the list of tensors to a single tensor with torch.stack
    train_data_tensor = torch.stack(train_data).type(torch.float32)
    test_data_tensor = torch.stack(test_data).type(torch.float32)
    val_data_tensor = torch.stack(val_data).type(torch.float32)

    # Create tensors for the labels
    train_labels = torch.tensor(train_labels).unsqueeze(1).float()
    test_labels = torch.tensor(test_labels).unsqueeze(1).float()
    val_labels = torch.tensor(val_labels).unsqueeze(1).float()

    # Call a function that dills the tensors
    dill_tensors(train_data_tensor, train_labels, test_data_tensor, test_labels, val_data_tensor, val_labels)

    # Delete the tensors to free up memory
    del train_data
    del test_data
    del val_data
    del train_labels
    del test_labels
    del val_labels

    # HYPERPARAMETERS
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 16


    # Load the dilled tensors
    train_data_tensor, train_labels, test_data_tensor, test_labels, val_data_tensor, val_labels = load_dilled_tensors()




    # Import Tensorset and DataLoader
    from torch.utils.data import TensorDataset, DataLoader

    # Create a TensorDataset
    train_dataset = TensorDataset(train_data_tensor, train_labels)
    test_dataset = TensorDataset(test_data_tensor, test_labels)
    val_dataset = TensorDataset(val_data_tensor, val_labels)

    # Create a DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Print amount of batches
    print(f"Amount of train batches: {len(train_loader)}",
          f"Amount of test batches: {len(test_loader)}",
          f"Amount of val batches: {len(val_loader)}",
          sep="\n", end="\n\n")


    # Clean up the memory allocated for the tensors
    del train_data_tensor
    del test_data_tensor
    del val_data_tensor
    del train_labels
    del test_labels
    del val_labels

    import gc
    gc.collect()

    # Define the model
    import Project.Model.Custom_CNN
    model = Custom_CNN.CNN_model(image_shape=(256, 256))

    # Import torchinfo summary
    from torchinfo import summary

    summary(model, input_size=(BATCH_SIZE, 1, 256, 256))

    # Define BinaryCrossEntropyLoss and Adam optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    train_losses, val_losses, val_accuracies = train_validate_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)

    test_model(model, test_loader, criterion)

    plot_train_val_test_metrics(train_losses, val_losses,
                                val_accuracies, EPOCHS)

if __name__ == "__main__":
    run()