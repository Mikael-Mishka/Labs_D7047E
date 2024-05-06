import torch
import os
import json
import io
import pickle

# Set Keras backend to TensorFlow
os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
from keras import backend as K  # For backend manipulation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from Data_preprocess import caption_preprocess, word_embedding, generate_results
from Model import multimodal_model


# Main task: Preprocess data, define model, and train
def Task1():
    # Run data preprocessing
    caption_preprocess()  # Extract and preprocess text and images
    
    max_length = 35
    truncation_length = 50
    end_token_idx = 15
    
    image_train, image_test, target_caption_train, input_caption_train, target_caption_test, input_caption_test, input_caption_pred, emb_mat = word_embedding(max_length)  # Load preprocessed data
    
    # Model parameters
    image_shape = (224, 224, 3)
    vocab_size, emb_dim = np.shape(emb_mat)
    
    target_caption_train_one_hot = to_categorical(target_caption_train, num_classes=vocab_size)
    target_caption_test_one_hot = to_categorical(target_caption_test, num_classes=vocab_size)
    
    # Define multimodal model
    model = multimodal_model(image_shape, max_length, vocab_size, emb_dim, emb_mat, end_token_idx, truncation_length)

    # Define callbacks for training
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')
    model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
    
    # Train the model
    history = model.fit(
        [image_train, input_caption_train],  # Adjust inputs to match model's expected inputs
        target_caption_train_one_hot,
        epochs=10,  # Modify based on the requirements
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint],
        verbose=True
    )
    
    # Store the history
    with open('./trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
        
    # To retrieve the history, write these lines
    #with open('./trainHistoryDict', "rb") as file_pi:
    #   history = pickle.load(file_pi)
    
    # Test the model
    
    model.load_weights('best_model.keras')
    
    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate([image_test, input_caption_test], target_caption_test_one_hot, batch_size=32)
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for 3 samples")
    out_file = io.open("./word_map.json", "r", encoding="utf-8-sig")
    word_map = json.load(out_file)
    out_file.close()
    test_images = image_test[:3]
    pred_caption = input_caption_pred[:3]
    prediction = model.predict([test_images, pred_caption], **dict(inference=True))
    
    #reverse_word_map = {v["Rep"]: k for k, v in word_map.items()}
    
    print('Images predicted:', test_images)
    print('Predictions:', prediction)
    

# Entry point for the script
def main():
    Task1()

if __name__ == "__main__":
    main()