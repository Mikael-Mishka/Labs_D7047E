import torch

# Pathlib and os
import pathlib
import os

# Import the Pre-made model.py
from Project.Model.Pre_made_Model import OUR_Resnet18

# Import matplotlib
import matplotlib.pyplot as plt

# Import f1_score from torcheval.metrics
from torcheval.metrics.functional import multiclass_f1_score

# Summary
from torchinfo import summary

# Import the serialized dataset
from Project.Preprocessing.SerializeData import load_serialized_data

# Import TensorDataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader


def test_resnet18(model, test_loader, loss_fn):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loss = 0.0
    test_accuracy = 0.0
    f1_macro_score = 0.0
    f1_weighted_score = 0.0

    model.eval()

    with torch.no_grad():
        for i, (image, labels) in enumerate(test_loader):
            image = image.to(device)
            labels = labels.squeeze(1).type(torch.LongTensor).to(device)

            outputs = model(image)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

            # Calculate test accuracy
            _, predicted = torch.max(outputs, 1)
            test_accuracy += (predicted == labels).sum().item()/len(labels)

            # Calculate F1 scores
            f1_macro_score += multiclass_f1_score(predicted, labels, num_classes=2, average="macro")
            f1_weighted_score += multiclass_f1_score(predicted, labels, num_classes=2, average="weighted")


    print(f"Test loss: {test_loss/len(test_loader)}",
          f"Test accuracy: {test_accuracy/len(test_loader)}",
          f"F1 macro score: {f1_macro_score/len(test_loader)}",
          f"F1 weighted score: {f1_weighted_score/len(test_loader)}",
          sep="\n")


def train_resnet18(model, train_loader, val_loader,
                   loss_fn, optimizer, fine_tuning_EPOCHS):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save the metrics for the fine-tuning
    fine_tuning_metrics = {
        "train_losses": [],
        "val_losses": [],
        "val_accuracies": [],
        "f1_macro": [],
        "f1_weighted": []
    }

    best_val_loss = 999999

    for epoch in range(fine_tuning_EPOCHS):

        train_loss = 0.0

        # Training
        model.train()
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.squeeze(1).type(torch.LongTensor).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate fine-tuning loss
            train_loss += loss.item()

        # Validation
        val_loss = 0.0
        val_accuracy = 0.0

        f1_macro = 0.0
        f1_weighted = 0.0

        model.eval()

        with torch.no_grad():
            for i, (image, labels) in enumerate(val_loader):
                image = image.to(device)
                labels = labels.squeeze(1).type(torch.LongTensor).to(device)

                outputs = model(image)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                # Calculate fine-tuning accuracy
                _, predicted = torch.max(outputs, 1)
                val_accuracy += (predicted == labels).sum().item()/len(labels)

                # Calculate f1 scores
                f1_macro += multiclass_f1_score(predicted, labels, num_classes=2, average="macro")
                f1_weighted += multiclass_f1_score(predicted, labels, num_classes=2, average="weighted")


        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, "best_resnet_finetune_model.pth")

        # Save the metrics
        fine_tuning_metrics["train_losses"].append(train_loss/len(train_loader))
        fine_tuning_metrics["val_losses"].append(val_loss/len(val_loader))
        fine_tuning_metrics["val_accuracies"].append(val_accuracy/len(val_loader))
        fine_tuning_metrics["f1_macro"].append(f1_macro/len(val_loader))
        fine_tuning_metrics["f1_weighted"].append(f1_weighted/len(val_loader))


        print(f"\nEpoch: {epoch+1}/{fine_tuning_EPOCHS}",
              f"Validation loss: {val_loss/len(val_loader)}",
              f"Validation accuracy: {val_accuracy/len(val_loader)}",
              f"Train loss: {train_loss/len(train_loader)}\n",
              sep="\n")

    return fine_tuning_metrics


def plot_fine_tuning_metrics(fine_tuning_metrics: dict[str, list],
                             fine_tuning_EPOCHS: int):

    # Create plots for train loss, val loss and val accuracy
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(fine_tuning_EPOCHS), fine_tuning_metrics["train_losses"], label="Train loss")
    plt.plot(range(fine_tuning_EPOCHS), fine_tuning_metrics["val_losses"], label="Val loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(fine_tuning_EPOCHS), fine_tuning_metrics["val_accuracies"], label="Val accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()

    # Force the stdout to capture the metrics
    print(f"\nFine-tuning metrics: {fine_tuning_metrics}\n")




def main():
    """
    We want to try to implement feature extraction using the pre-trained model
    and get test results for the model.
    """

    # Get the serialized objects
    (train_labels,
     test_labels,
     val_labels,
     train_data_tensor,
     test_data_tensor,
     val_data_tensor) = load_serialized_data()

    # Define HYPERPARAMETERS
    fine_tuning_EPOCHS = 1
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 16

    # Absolute path of parent directory
    parent_dir = pathlib.Path(__file__).parent.parent.absolute()

    # Get saved model path
    saved_model_path = parent_dir / "Model" / "ammonia_resnet18.pth"

    # Load the model
    model = torch.load(saved_model_path)

    # Define loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Prepare the Tensordataset
    train_data = TensorDataset(train_data_tensor, train_labels)
    test_data = TensorDataset(test_data_tensor, test_labels)
    val_data = TensorDataset(val_data_tensor, val_labels)

    # Prepare the DataLoader
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # Print summary of the model
    summary(model)

    # Change the conv1 layer to accept 1 channel in model.resnet18_model
    model.resnet18_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    device = None

    if torch.cuda.is_available():
        device = torch.device("cuda")

    model.to(device)

    # Fine-tuning the model
    train_resnet18(model, train_loader, val_loader,
                   loss_fn, optimizer, fine_tuning_EPOCHS)

    test_resnet18(model, test_loader, loss_fn)


if __name__ == "__main__":
    main()