import pathlib
import pickle
import torch
from EncoderTransformer import Transformer
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torchinfo import summary
import math
# import TensorDataset
from torch.utils.data import DataLoader, TensorDataset


def train_model(model, train_loader, validation_loader, epochs, optimizer, loss_fn, device):

    # Add step scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=False)

    train_losses = []
    validation_losses = []
    validation_accuracies = []

    loss_fn = loss_fn.to(device=device)

    best_loss = torch.inf

    for epoch in range(epochs):

        current_train_loss = 0
        current_validation_loss = 0
        current_validation_accuracy = 0

        model.train()

        for i, (data, target) in enumerate(train_loader):

            data = data.to(device=model.device)
            target = target.to(device=model.device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()

            optimizer.step()

            current_train_loss += loss.item()

        model.eval()

        with torch.no_grad():
            for i, (data, target) in enumerate(validation_loader):
                output = model.forward(data)

                output = output.to(device=model.device)
                target = target.to(device=model.device)

                loss = loss_fn(output, target)

                # Track the loss
                current_validation_loss += loss.item()

                # Track the accuracy
                _, predicted = torch.max(output, 1)

                # send predicted to cuda
                predicted = predicted.to(device=model.device)

                correct = (predicted == target).sum().item()
                total = target.size(0)

                current_validation_accuracy += correct / total

        current_train_loss /= len(train_loader)
        current_validation_loss /= len(validation_loader)
        current_validation_accuracy = current_validation_accuracy / len(validation_loader) * 100

        # Step the scheduler
        scheduler.step(current_validation_loss, epoch)

        if current_validation_loss < best_loss:
            best_loss = best_loss

            # Save the transformer model
            with open("best_transformer.pth", "wb") as f:
                torch.save(model, f)

        # Print epoch information
        print(f"Epoch: {epoch}, Train Loss: {current_train_loss}, Validation Loss: {current_validation_loss}, Validation Accuracy: {current_validation_accuracy}")

        # Append losses and accuracy
        train_losses.append(current_train_loss)
        validation_losses.append(current_validation_loss)
        validation_accuracies.append(current_validation_accuracy)


    return train_losses, validation_losses, validation_accuracies


def run():
    # Define HYPERPARAMETERS
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.001

    # Unpickle the data
    abs_path = pathlib.Path(__file__).parent.absolute()

    # Remove "/Task_2" from the path
    train_pickle_path = abs_path.parent.joinpath("data", "picklies", "training_tensors_p0.pkl")

    with open(train_pickle_path, "rb") as f:
        unpickled_train_data = pickle.load(f)

    # Find validation path
    validation_pickle_path = abs_path.parent.joinpath("data", "picklies", "validation_tensors_p0.pkl")

    with open(validation_pickle_path, "rb") as f:
        unpickled_validation_data = pickle.load(f)

    training_data, training_labels = unpickled_train_data
    validation_data, validation_labels = unpickled_validation_data

    # Adjust the path to the vocab_size.pkl file
    vocab_size_path = abs_path.parent.joinpath("data", "picklies", "vocab.pkl")

    with open(vocab_size_path, "rb") as f:
        unpickled_vocab_list = pickle.load(f)

    vocab_size = len(unpickled_vocab_list)

    device = torch.device("cuda:0")

    # Dummy data for testing
    dummy_data = training_data[0].to(device=device)

    # Define ntokens
    ntokens = training_data.shape[1]

    dummy_data = torch.randint(low=0, high=5, size=(BATCH_SIZE, ntokens))
    print(dummy_data.shape)

    # Define the TensorDatasets
    training_dataset = TensorDataset(training_data, training_labels)
    validation_dataset = TensorDataset(validation_data, validation_labels)

    # Define the DataLoaders
    training_data = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_data = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    import gc
    print(f"Garbage collector: {gc.collect()}")

    # Define the model
    model = Transformer(vocab_size=vocab_size, ntokens=ntokens, d_model=128, nhead=8, num_encoder_layers=6, dim_feedforward=512, dropout=0.1, device=device, BATCH_SIZE=BATCH_SIZE)
    summary(model, input_data=dummy_data)
    model(dummy_data)

    # Import perf_counter
    from time import perf_counter

    start_time = perf_counter()
    train_losses, validation_losses, validation_accuracies = train_model(model, training_data, validation_data, EPOCHS, torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5), nn.CrossEntropyLoss(), device)
    end_time = perf_counter()


    best_validation_accuracy = max(validation_accuracies)
    best_validation_loss = min(validation_losses)

    print(f"----- Training finished -----",
          f"1. Best validation accuracy: {best_validation_accuracy}",
          f"2. Best epoch: {validation_accuracies.index(max(validation_accuracies))}",
          f"3. Best Perplexity: {math.e**best_validation_loss}",
          f"4. Time to train {EPOCHS} Epochs for Encoder transformer {end_time-start_time}", sep="\n")

if __name__ == "__main__":
    run()