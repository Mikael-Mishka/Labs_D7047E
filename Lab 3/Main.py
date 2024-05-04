# Import torch
import torch

# Import transforms
from torchvision import transforms

import numpy as np
import os
os.environ["KERAS_BACKEND"] = "torch"

#from keras.applications import efficientnet

# Device configuration
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from Data_preprocess import caption_preprocess, word_embedding
from Model import model

def Task1():
    Model = model(input_shape = (224,224,3)) # shape to be modified
    Model.summary()

# This is the entry point of the program
def main():
    if True:
        caption_preprocess()
        X_train, X_test, Y_train, Y_test, emb_mat = word_embedding()
    Task1()

if __name__ == "__main__":
    main()