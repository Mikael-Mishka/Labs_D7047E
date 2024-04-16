import os

from torch import nn
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
import re

# import vectorizer from the data file. It would make sense
# for us to the same as was to prepare the data.
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords

# LSTM for analysing text sequences
# and outputing it to a linear layer
# as a negative or positive review
class SimpleChatbot(nn.Module):

    def __init__(self, input_size, num_classes):
        super().__init__()

        hidden_dim = 24

        dummy = torch.Tensor([i for i in range(input_size)])
        embed_size = nn.Linear(input_size, hidden_dim)(dummy)
        flatten_embed_size = embed_size.flatten().shape[0]

        print(flatten_embed_size)


        self.internal_model = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.Flatten(),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(flatten_embed_size, 200),
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


    def train_model(self, train_loader, validation_loader, epochs, optimizer, loss_fn):

        for epoch in range(epochs):

            # Track loss, validation accuracy, and validation loss
            train_loss = 0
            validation_accuracy = 0
            validation_loss = 0


            for i, (data, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.forward(data)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()

                # Track the loss
                train_loss += loss.item()

            # Track the validation accuracy and loss
            with torch.no_grad():
                for i, (data, labels) in enumerate(validation_loader):
                    output = self.forward(data)
                    validation_loss += loss_fn(output, labels).item()

                    _, predicted = torch.max(output, 1)
                    validation_accuracy += (predicted == labels).sum().item()/len(labels)


            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss / len(train_loader)} - Validation Loss: {validation_loss / len(validation_loader)} - Validation Accuracy: {validation_accuracy / len(validation_loader)}")

