import torch
import torch.nn as nn


def train_model(model, train_loader, validation_loader, epochs, loss_fn, optimizer):
    # best_loss
    best_loss = 1e9

    # Training loop metrics
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=False, factor=0.1)

    for epoch in range(epochs):

        current_training_loss = 0
        current_validation_loss = 0
        current_validation_accuracy = 0

        # Training loop
        for i, (data, labels) in enumerate(train_loader):

            # Reset the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Calculate the loss
            loss = loss_fn(output, labels)

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

            # Update the current training loss
            current_training_loss += loss.item()

        # Validation loop

        with torch.no_grad():

            for i, (data, labels) in enumerate(validation_loader):


                # Forward pass
                output = model(data)

                # Calculate the loss
                loss = loss_fn(output, labels)

                # Update the current validation loss
                current_validation_loss += loss.item()

                # Calculate correct and total
                _, predicted = torch.max(output, 1)
                correct = (predicted == labels).sum().item()
                total = labels.size(0)

                # Update the current validation accuracy
                current_validation_accuracy += correct / total

        # Calculate the averages of the metrics
        current_training_loss /= len(train_loader)
        current_validation_loss /= len(validation_loader)
        current_validation_accuracy = current_validation_accuracy / len(validation_loader) * 100

        # Step the scheduler
        scheduler.step(current_validation_loss)

        # Append the metrics to the lists
        train_losses.append(current_training_loss)
        val_losses.append(current_validation_loss)
        val_accuracies.append(current_validation_accuracy)

        # Print the metrics
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {current_training_loss} - Validation Loss: {current_validation_loss} - Validation Accuracy: {current_validation_accuracy}")

        if current_validation_loss < best_loss:
            best_loss = best_loss

            # Save the model
            with open("best_ann_model.pth", "wb") as f:
                torch.save(model.state_dict, f)

    return train_losses, val_losses, val_accuracies, model


class ModifiedSimpleChatbot(nn.Module):

    def __init__(self, input_size, d_model, vocab_size, num_classes):
        super().__init__()


        self.internal_model = nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            nn.Flatten(),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(d_model * input_size, 200),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(200, 480),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(480, 100),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        return self.internal_model(x)

