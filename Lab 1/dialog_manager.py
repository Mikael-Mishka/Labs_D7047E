# Directory Task_2, Task_2.py imported

import torch
from Task_1.SimpleChatbot import SimpleChatbot
import re
from data import data_loading_task_2_3, data_loading_code
from nltk.corpus import stopwords
from nltk import word_tokenize
import torch.nn as nn
import random

# import TensorDataset
from torch.utils.data import DataLoader, TensorDataset

class DialogManager():

    def __init__(self):

        # Define HYPERPARAMETERS
        self.EPOCHS, self.BATCH_SIZE, self.LEARNING_RATE = 30, 64, 1e-3

        # Task one review types
        NEGATIVE = 0
        POSITIVE = 1

        # Task 1 Simple Chatbot possible responses
        self.task_1_responses: dict[1, list[str]] = {
            POSITIVE: ["Fantastic! We are glad you liked your purchase. Do you have any more reviews to submit?",
                       "Awesome! You happy we happy is our motto. Do you happen to have any more reviews to tell me about today?",
                       "Great to hear you liked your purchase! Anything you have any more reviews to submit?"],

            NEGATIVE: ["Sorry you did not like your purchase. Do you have any more reviews to submit?",
                       "Thank you for the review. We will try to improve in the future! Would you like to leave another review?",
                       "We are sorry to hear that you did not like your purchase. Any other purchases to review today?"]
        }


    def prepare_user_input(user_inp: str, vectorizer):
        # Preprocess user input
        user_input = user_inp.lower()
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

    def task_1(self):

        # Prepare the data
        (train_x_tensor,
         train_y_tensor,
         validation_x_tensor,
         validation_y_tensor,
         vocab_size, word_vectorizer) = data_loading_code.get_data()

        # Define the Simple Chatbot
        self.simple_chatbot = SimpleChatbot(vocab_size, 2)

        # Create the Datasets
        train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
        validation_dataset = TensorDataset(validation_x_tensor, validation_y_tensor)

        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=self.BATCH_SIZE, shuffle=True)

        # Define the optimizer and loss function
        optimizer = torch.optim.Adam(self.simple_chatbot.parameters(), lr=self.LEARNING_RATE)
        loss_fn = nn.CrossEntropyLoss()

        # Train the Simple chatbot
        self.simple_chatbot.train_model(train_loader, validation_loader, self.EPOCHS, optimizer, loss_fn)

        running: bool = True


        print(f"<ReviewBot> Welcome! Type 'exit' if you don't want to leave more reviews.")
        while running:
            user_input = input("<User> ").lower()
            if user_input == "exit":
                running = False
            else:
                user_tensor = DialogManager.prepare_user_input(user_input, word_vectorizer)
                output = self.simple_chatbot(user_tensor)

                output_probabilities = torch.nn.functional.softmax(output, dim=1)

                prediction = torch.argmax(output_probabilities).item()

                print("<ReviewBot> " + self.task_1_responses[prediction][random.randint(0, 2)])

        return self.simple_chatbot


