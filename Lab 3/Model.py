import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Embedding, Dense, Activation, MaxPool1D, Input, LSTM, Dropout, Input, Activation, Add, Layer, \
    MaxPooling2D, Conv2D, Flatten, Concatenate
from keras.optimizers import Adam
from keras.regularizers import l2
import numpy as np

class CustomMultiModalLayer(Layer):
    def __init__(self, emb_dim, emb_mat, vocab_size, input_shape, end_token_idx, truncation_length, *args, **kwargs):
        super(CustomMultiModalLayer, self).__init__(*args, **kwargs)
        
        # Image layers
        self.CNN = CustomMultiModalLayer.image_block(input_shape)
        
        # Text layers
        self.emb1 = Embedding(input_dim=vocab_size, output_dim=emb_dim, weights=[emb_mat], trainable=False)
        
        # LSTM layers
        self.lstm1 = LSTM(emb_dim // 2, return_sequences=True)
        self.lstm2 = LSTM(emb_dim, return_sequences=False)
        
        # Output layer
        self.output_layer = Dense(vocab_size, activation='softmax')
        
        # End token and max length
        self.end_token_idx = end_token_idx
        self.truncation_length = truncation_length
    
    def call(self, inputs, training=None, **kwargs):
        # Unpack inputs
        img_input, text_input = inputs
        
        # Extract image features
        img_features = self.CNN(img_input)
        
        if training:
            # Get text features via embedding
            text_features = self.emb1(text_input)
            # Concatenate image features with each text timestep
            combined_features = Concatenate(axis=-1)([tf.tile(tf.expand_dims(img_features, 1), [1, tf.shape(text_features)[1], 1]), text_features])
            
            # Process through LSTM layers
            x = self.lstm1(combined_features)
            x = Dropout(0.2)(x)
            x = self.lstm2(x)
            x = Dropout(0.2)(x)
            x = self.output_layer(x)
            return x
        elif not training and not kwargs.get('inference'):
            # Get text features via embedding
            text_features = self.emb1(text_input)
            # Concatenate image features with each text timestep
            combined_features = Concatenate(axis=-1)([tf.tile(tf.expand_dims(img_features, 1), [1, tf.shape(text_features)[1], 1]), text_features])
            
            # Process through LSTM layers
            x = self.lstm1(combined_features)
            x = self.lstm2(x)
            x = self.output_layer(x)
            return x
        elif not training and kwargs.get('inference'):
            # Inference logic
            outputs = []
            current_token = text_input[:, 0:1]  # starting token (e.g., <start>)
            
            for _ in range(self.truncation_length):
                # Get text embedding for the current token
                current_emb = self.emb1(current_token)
                # Concatenate image features with the current token's embedding
                current_features = Concatenate(axis=-1)([img_features, tf.squeeze(current_emb, axis=1)])
                # Reshape to maintain batch dimension
                current_features = tf.expand_dims(current_features, 1)
                
                # Pass through LSTM layers
                x = self.lstm1(current_features)
                x = Dropout(0.2)(x)
                x = self.lstm2(x)
                x = Dropout(0.2)(x)
                x = self.output_layer(x)  # softmax output
                
                predicted_token = tf.argmax(x, axis=-1)
                outputs.append(predicted_token)
                
                # Stop if the end token is predicted
                if tf.reduce_all(predicted_token == self.end_token_idx):
                    break
                
                # Set the next input token to the predicted one
                current_token = predicted_token
            
            return tf.stack(outputs, axis=1)  # Stack outputs to maintain the sequence dimension
    
    @staticmethod
    def image_block(input_shape):
        inputs = Input(shape=input_shape)
        x = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.003))(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.003))(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.003))(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        return Model(inputs=inputs, outputs=x)

# Multimodal model that combines image and text branches using Concatenate
def multimodal_model(image_shape, max_length, vocab_size, emb_dim, emb_mat, end_token_idx, truncation_length):
    
    image_input = Input(shape=(image_shape,))
    text_input = Input(shape=(max_length,))
    
    custom_layer = CustomMultiModalLayer()(emb_dim, emb_mat, vocab_size, image_shape, end_token_idx, truncation_length)

    # Define the final multimodal model
    model = Model(inputs=[image_input, text_input], outputs=custom_layer)
 
    model.summary()
    
    # Compile the model
    model.compile(loss='crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    return model

# Usage example (outside this class):
# Define image_size, input_shape, emb_dim, emb_mat, vocab_size, end_token_idx, max_length.
# layer = CustomMultiModalLayer(emb_dim, emb_mat, vocab