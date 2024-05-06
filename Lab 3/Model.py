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
        self.args = (emb_dim, emb_mat, vocab_size, input_shape, end_token_idx, truncation_length, args, kwargs)
        
        # Image layers
        self.CNN = CustomMultiModalLayer.image_block(input_shape)
        
        # Text layers
        self.emb1 = Embedding(input_dim=vocab_size, output_dim=emb_dim, weights=[emb_mat], trainable=False, mask_zero=True)
        
        # LSTM layers
        self.lstm1 = LSTM(emb_dim // 2, return_sequences=True)
        self.lstm2 = LSTM(emb_dim, return_sequences=False)
        
        # Output layer
        self.output_layer = Dense(vocab_size, activation='softmax')
        
        # End token and max length
        self.end_token_idx = end_token_idx
        self.truncation_length = truncation_length
    
    def call(self, inputs, training=None):
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
        elif not training and len(text_input.shape) > 1 and text_input.shape[1] > 1: # Evaluate with full sequence
            # Get text features via embedding
            text_features = self.emb1(text_input)
            # Concatenate image features with each text timestep
            combined_features = Concatenate(axis=-1)([tf.tile(tf.expand_dims(img_features, 1), [1, tf.shape(text_features)[1], 1]), text_features])

            # Process through LSTM layers
            x = self.lstm1(combined_features)
            x = self.lstm2(x)
            x = self.output_layer(x)
            return x
        elif not training and text_input.shape[0] in [None, 1]: # Inference runs with a sequence length of 1
            if text_input.shape[0] is None:
                text_input = tf.expand_dims(text_input, 1)

            current_token = text_input
            current_token = tf.cast(current_token, tf.dtypes.int64)
            print(f"Starting dtype: {current_token.dtype}, shape: {current_token.shape}")

            outputs = tf.TensorArray(dtype=tf.int64, size=self.truncation_length, dynamic_size=True, clear_after_read=False)
            for i in tf.range(self.truncation_length):
                tf.autograph.experimental.set_loop_options(
                    shape_invariants=[
                        (current_token, tf.TensorShape([None, 1]))
                    ]
                )

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

                predicted_token = tf.argmax(x, axis=-1, output_type=tf.dtypes.int64)
                outputs = outputs.write(i, predicted_token)

                # Stop if the end token is predicted
                cond = tf.reduce_all(predicted_token == self.end_token_idx)
                if tf.equal(cond, True):
                    break

                predicted_token = tf.expand_dims(predicted_token, 1)
                current_token = tf.cast(predicted_token, tf.dtypes.int64)
                print(f"Current dtype: {current_token.dtype}, shape: {current_token.shape}")

            return outputs.stack("output")

        raise ValueError('Invalid mode')
    
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

    def get_config(self):
        config = {
            'init_params': self.args,
            'image_model_config': self.CNN.get_config() if self.CNN else None
        }
        base_config = super(CustomMultiModalLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        image_model_config = config.pop('image_model_config')

        args = config['init_params'].pop('args')
        kwargs = config['init_params'].pop('kwargs')
        inst = cls(*args, **(kwargs | config['init_params']))
        if image_model_config:
            image_model = Model.from_config(image_model_config)
        else:
            image_model = None
        inst.CNN = image_model

        return inst

# Multimodal model that combines image and text branches using Concatenate
def multimodal_model(image_shape, max_length, vocab_size, emb_dim, emb_mat, end_token_idx, truncation_length):
    custom_layer = CustomMultiModalLayer(emb_dim, emb_mat, vocab_size, image_shape, end_token_idx, truncation_length)

    # Define input layers
    img_input = tf.keras.Input(shape=image_shape, name='image_input')
    text_input = tf.keras.Input(shape=(max_length,), name='text_input')

    # Connect inputs to the layer
    output = custom_layer([img_input, text_input])

    # Create model
    model = tf.keras.Model(inputs=[img_input, text_input], outputs=output)

    model.summary()
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    return model

# Usage example (outside this class):
# Define image_size, input_shape, emb_dim, emb_mat, vocab_size, end_token_idx, max_length.
# layer = CustomMultiModalLayer(emb_dim, emb_mat, vocab