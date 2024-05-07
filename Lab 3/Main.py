import os
import json
import io
import threading
import time

import dill


import numpy as np
import pathlib

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn
from torch.optim import Adam
from torchvision.transforms import transforms

from Data_preprocess import caption_preprocess, word_embedding
from Model import multimodal_model
import gc as garbage_man


# Main task: Preprocess data, define model, and train
def Task1():
    # Run data preprocessing
    # caption_preprocess()  # Extract and preprocess text and images
    
    max_length = 35
    truncation_length = 50
    end_token_idx = 15

    data_tuple_path = pathlib.Path('./data_tuple.dill')
    if data_tuple_path.exists():
        with open(data_tuple_path, 'rb') as file:
            image_train, image_test, target_caption_train, input_caption_train, target_caption_test, input_caption_test, input_caption_pred, emb_mat = dill.load(file)
    else:
        data_tuple = word_embedding(max_length)  # Load preprocessed data
        garbage_man.collect()
        with open(data_tuple_path, 'wb') as file:
            dill.dump(data_tuple, file)

        image_train, image_test, target_caption_train, input_caption_train, target_caption_test, input_caption_test, input_caption_pred, emb_mat = data_tuple
        del data_tuple
        garbage_man.collect()

    # Model parameters
    image_shape = (3, 224//2, 224//2)
    vocab_size, emb_dim = np.shape(emb_mat)
    
    # Define multimodal model
    model = multimodal_model(max_length, vocab_size, image_shape, emb_dim, emb_mat, end_token_idx, truncation_length)
    model.remove_softmax()

    training_loader = DataLoader(
        TensorDataset(torch.tensor(image_train), torch.tensor(input_caption_train), torch.tensor(target_caption_train)),
        batch_size=32,
        shuffle=True
    )
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    transform = transforms.Compose([
        transforms.Resize((224//2, 224//2)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization
    ])

    from torchinfo import summary
    print(summary(model, input_size=(32, 3, 224//2, 224//2), device='cpu'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    start_time = time.time()
    # Training loop
    for epoch in range(100):
        model.train()

        total = 0
        e_loss = 0
        correct = 0
        for i, (image, input_caption, target_caption) in enumerate(training_loader):
            optimizer.zero_grad()
            # Image is W, H, C
            image = image.permute(0, 3, 1, 2)
            image = transform(image.float())
            image = image.to(device)
            input_caption = input_caption.to(device)
            target_caption = target_caption.to(device)
            target_caption = target_caption.view(-1)
            total += target_caption.size(0)

            output = model(image, input_caption, 'train')
            loss = criterion(output[:, -1, :], target_caption)

            prediction = torch.argmax(output[:, -1, :], dim=1)
            correct += torch.sum(prediction == target_caption).item()
            e_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            optimizer.step()
            if time.time() - start_time > 1/2 or i == len(training_loader) - 1:
                print(f"Epoch {epoch}, Batch {i}, Loss: {e_loss / total}, Accuracy: {correct / total}", end='\r' if i < len(training_loader) - 1 else '\n', flush=True)
                start_time = time.time()

    test_tensor = torch.tensor(image_test)
    pred_data = DataLoader(
        TensorDataset(test_tensor, torch.tensor([input_caption_pred]*test_tensor.size(0))),
        batch_size=32,
        shuffle=False
    )

    model.eval()

    predictions = []

    for i, (image, input_caption) in enumerate(pred_data):
        image = image.permute(0, 3, 1, 2)
        image = transform(image.float())
        image = image.to(device)
        input_caption = input_caption.to(device)
        output = model(image, input_caption, 'infer')
        predictions.append(output)

    print(*predictions, sep='\n' * 2)
# Entry point for the script
def main():
    Task1()

if __name__ == "__main__":
    main()