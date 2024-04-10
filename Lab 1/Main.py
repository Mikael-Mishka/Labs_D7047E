import os

from data import data_loading_code
from dialogue_manager import dialogue_manager
from torch import nn
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
import re
import numpy as np

# import vectorizer from the data file. It would make sense
# for us to the same as was to prepare the data.
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords

# LSTM for analysing text sequences
# and outputing it to a linear layer
# as a negative or positive review
class SimpleFNN(nn.Module):

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


            #print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss / len(train_loader)} - Validation Loss: {validation_loss / len(validation_loader)} - Validation Accuracy: {validation_accuracy / len(validation_loader)}")

def chat_bot_response(predicted):

    positive = 1
    negative = 0

    chat_responses: dict[int, tuple[str, str]] = {
        positive: ("Awesome! That makes me happy to hear! Do you have another review to make?",
                          "I am glad you had a good experience. Would you like to provide another review?"),

        negative: ("I am sorry to hear you had a bad experience. Would you like to provide another review?",
                         "I apologize for the inconvinience for you. Would you like to provide another review?")
    }

    # Return any of the two available responses per review
    return chat_responses[predicted.item()][random.randint(0, 1)]


def prepare_user_input(user_input, vectorizer):
    # Preprocess user input
    user_input = user_input.lower()
    user_input = re.sub(r'[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', user_input)  # remove emails
    user_input = re.sub(r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', user_input)  # remove IP address
    user_input = re.sub(r'[^\w\s]', '', user_input)  # remove special characters
    user_input = re.sub('\d', '', user_input)  # remove numbers
    word_tokens = word_tokenize(user_input)
    filtered_sent = [w for w in word_tokens if not w in stopwords.words('english')]
    processed_input = " ".join(filtered_sent)

    # Vectorize user input using the provided TF-IDF vectorizer
    user_tfidf = vectorizer.transform([processed_input])

    # Convert to PyTorch tensor
    user_tensor = torch.tensor(user_tfidf.toarray(), dtype=torch.float32)

    return user_tensor



def main():
    train_x_tensor, train_y_tensor, validation_x_tensor, validation_y_tensor, vocab_size, word_vectorizer = data_loading_code.get_data()

    # HYPERPARAMETERS
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 32

    # Prepare the train, validation loaders.
    train_loader = DataLoader(TensorDataset(train_x_tensor, train_y_tensor), batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(TensorDataset(validation_x_tensor, validation_y_tensor), batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleFNN(vocab_size, 2)

    model.train_model(train_loader, validation_loader, EPOCHS, torch.optim.Adam(model.parameters(), lr=LEARNING_RATE), nn.CrossEntropyLoss())

    model.eval()

    print(f"\nType 'exit' to end the conversation.",
          f"Otherwise type your review and the chat bot will respond accordingly.\n",
          sep="\n")

    print("ReviewBot: Hello! How was your experience with the product?")

    while True:
        text = input("User: ")
        if text == "exit":
            break
        user_prompt = prepare_user_input(text, word_vectorizer)

        output = model(user_prompt)
        _, predicted = torch.max(output, 1)

        print("ReviewBot:", chat_bot_response(predicted), sep=" ")
        
        # The following lines count and display the answers of the network (just an example, can be used during testing)
        prev_answers = []
        answer_cnt = np.zeros(2)
        predicted_class = predicted[0].item()
        expected_class = 0 # placeholder, must be replaced by the label in a test context
        prev_answers, answer_cnt = dialogue_manager.answer_tracker(predicted_class, expected_class, prev_answers=prev_answers, answer_cnt=answer_cnt)
        dialogue_manager.display_answers(prev_answers, answer_cnt)




# Here is were Lab 1 starts
if __name__ == "__main__":
    main()
