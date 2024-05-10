# Imports path and file manipulation libraries
import os
import pathlib
from PIL import Image

# data manipulation libraries and torchvision
import torch
import torchvision.transforms as transforms

import dill


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


def run() -> None:
    (train_data,
     train_labels,
     test_data,
     test_labels,
     val_data,
     val_labels) = dataset_to_tensors(
        dataset_dir_path=pathlib.Path(r"C:\Users\mikluy-1\PycharmProjects\Labs_D7047E\Project\Pre-processing\DatasetResizedGrayscaled"),
        pickled_tensors_dir_path=pathlib.Path(r"C:\Users\mikluy-1\PycharmProjects\Labs_D7047E\Project\Pre-processing\Picklies")
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
    train_data_tensor = torch.stack(train_data)
    test_data_tensor = torch.stack(test_data)
    val_data_tensor = torch.stack(val_data)

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

    # Load the dilled tensors
    train_data_tensor, train_labels, test_data_tensor, test_labels, val_data_tensor, val_labels = load_dilled_tensors()

    print(f"Check shapes of loaded tensors",
            f"train_data_tensor.shape: {train_data_tensor.shape}",
            f"train_labels.shape: {train_labels.shape}",
            f"test_data_tensor.shape: {test_data_tensor.shape}",
            f"test_labels.shape: {test_labels.shape}",
            f"val_data_tensor.shape: {val_data_tensor.shape}",
            f"val_labels.shape: {val_labels.shape}",
            sep="\n")

    # Import Tensorset and DataLoader
    from torch.utils.data import TensorDataset, DataLoader

    # Create a TensorDataset
    train_dataset = TensorDataset(train_data_tensor, train_labels)

    # Create a DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Define BinaryCrossEntropyLoss
    import torch.nn as nn
    criterion = nn.BCELoss()

    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(246016, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid(),
    )

    # print one batch
    for data, labels in train_loader:
        output = model(data)


        print(criterion(output, labels))
        break



if __name__ == "__main__":
    run()
