import torch
import os

# Set Keras backend to TensorFlow
os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
from keras import backend as K  # For backend manipulation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from Data_preprocess import caption_preprocess, word_embedding
from Model import multimodal_model


# Main task: Preprocess data, define model, and train
def Task1():
    # Run data preprocessing
    #caption_preprocess()  # Extract and preprocess text and images
    X_train, X_test, Y_train, Y_test, emb_mat = word_embedding()  # Load preprocessed data
    
    # Model parameters
    image_input_shape = (224, 224, 3)
    vocab_size, emb_dim = np.shape(emb_mat)
    max_length = 40

    # Define multimodal model
    model = multimodal_model(image_input_shape, vocab_size, emb_dim, emb_mat, max_length)

    # Define callbacks for training
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')
    model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
    
    # target = target.unsqueeze(vocab_size)
    
    # Train the model
    model.fit(
        [X_train, Y_train],  # Adjust inputs to match model's expected inputs
        Y_train,
        epochs=10,  # Modify based on your requirements
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint],
        verbose=True
    )

# Entry point for the script
def main():
    Task1()

if __name__ == "__main__":
    main()