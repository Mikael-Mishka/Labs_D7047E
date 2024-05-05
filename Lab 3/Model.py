import keras
from keras.models import Model
from keras.layers import Embedding, Dense, Activation, MaxPool1D, Input, LSTM, Dropout, Input, Activation, Add, \
    MaxPooling2D, Conv2D, Flatten, Concatenate
from keras.optimizers import Adam
from keras.regularizers import l2
import numpy as np


def Convolution(input_tensor, filters):
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(0.003))(
        input_tensor)
    x = Dropout(0.1)(x)
    x = Activation('relu')(x)
    return x


def image_model(input_shape):
    inputs_images = Input((input_shape))
    conv_1 = Convolution(inputs_images, 32)
    maxp_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
    conv_2 = Convolution(maxp_1, 64)
    maxp_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
    conv_3 = Convolution(maxp_2, 128)
    maxp_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
    flatten = Flatten()(maxp_3)
    dense_1 = Dense(256, activation='relu')(flatten)
    return Model(inputs=inputs_images, outputs=dense_1)

# Define the language model (LSTM-based)
def text_model(vocab_size, emb_dim, emb_mat, max_length):
    inputs_language=Input(shape=((max_length, )))
    emb1 = Embedding(input_dim=vocab_size, output_dim=emb_dim, weights=[emb_mat], trainable=False)(inputs_language)
    dr1 = Dropout(0.2)(emb1)
    lstm1 = LSTM(128, return_sequences=True)(dr1)
    
    lstm2 = LSTM(256, return_sequences=True)(lstm1)  # Return only the final output
    return Model(inputs=inputs_language, outputs=lstm2)

# Multimodal model that combines image and text branches using Concatenate
def multimodal_model(image_input_shape, vocab_size, emb_dim, emb_mat, max_length):
    image_model_instance = image_model(image_input_shape)  # CNN model
    text_model_instance = text_model(vocab_size, emb_dim, emb_mat, max_length)  # LSTM model

    # Concatenate outputs from both branches
    combination = Add()([image_model_instance.output, text_model_instance.output])

    # Additional dense layers to process the combined features
    dense_2 = Dense(256, activation='relu')(combination)
    output_layer = Dense(vocab_size, activation='softmax')(dense_2)  # Output layer

    # Define the final multimodal model
    model = Model(inputs=[image_model_instance.input, text_model_instance.input], outputs=output_layer)
 
    model.summary()
    
    # Compile the model
    model.compile(loss='crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    return model



