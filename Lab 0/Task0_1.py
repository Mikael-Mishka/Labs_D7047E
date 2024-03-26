from torch import nn
import torch
import time


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNN(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # If cuda is available, use it
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, 3, (1, 1), 1, device=self.device),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3, (1, 1), 1, device=self.device),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 3, 3, (1, 1), 1, device=self.device),
            nn.LeakyReLU(),
        )

        conv_layers.to(self.device)

        # Dummy pass to get the ouput tensor size of the conv layers
        dummy = torch.randn(1, 3, 32, 32).to(self.device)

        # Get the output size of the conv layers
        conv_out = conv_layers(dummy)

        num_features = Flatten().forward(conv_out).shape[1]

        self.internal_model = nn.Sequential(
            conv_layers,
            Flatten(),
            nn.Linear(num_features, 100),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(100, 10),
        )

        self.internal_model.to(self.device)

    def forward(self, x):
        return self.internal_model(x)

    def train_model(self, train_loader, validation_loader, loss_fn, optimizer, num_epochs):

        # Here is the metrics we will be tracking
        best_accuracy = 0
        best_epoch = 0
        current_accuracy = 0
        current_train_loss = 0
        current_validation_loss = 0

        last_print = time.time()

        # Here is the main training loop
        for epoch in range(num_epochs):

            # Reset the metrics
            current_accuracy = 0
            current_train_loss = 0
            current_validation_loss = 0


            # Sets us in training mode
            self.train()
            # Training loop
            for i, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()

                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self(images)
                loss = loss_fn(outputs, labels)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                # Track the loss
                current_train_loss += loss.item()

                # Track the accuracy
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).sum().item()
                total = labels.size(0)
                current_accuracy += correct / total

            # Sets us in evaluation mode
            self.eval()

            # Validation loop
            with torch.no_grad():
                for i, (images, labels) in enumerate(validation_loader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass
                    outputs = self(images)
                    loss = loss_fn(outputs, labels)

                    # Track the loss
                    current_validation_loss += loss.item()

                    # Track the accuracy
                    _, predicted = torch.max(outputs, 1)
                    correct = (predicted == labels).sum().item()
                    total = labels.size(0)
                    current_accuracy += correct / total

            # Calculate the average loss and accuracy
            current_train_loss /= len(train_loader)
            current_validation_loss /= len(validation_loader)
            current_accuracy /= len(validation_loader)

            if time.time() - last_print > 15:

                last_print = time.time()
                # Print the results
                print(f"Epoch: {epoch}/{num_epochs}, Train Loss: {current_train_loss}, Validation Loss: {current_validation_loss}, Accuracy: {current_accuracy}")

            # Check if we have a new best accuracy
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_epoch = epoch
