from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn import Module, Conv2d, MaxPool2d, BatchNorm2d, Linear, LSTM, Embedding, Dropout, Sequential, LeakyReLU, ReLU, LogSoftmax, Softmax, Flatten
import numpy as np

# LSTM() returns tuple of (tensor, (recurrent state)): https://stackoverflow.com/a/64265525/22744854
class extract_tensor(Module):
    def forward(self, x):
        # Assume batch if it's 3d
        if len(x[0].shape) > 2:
            # Output shape (batch, sequence, features, hidden)
            tensor, _ = x
            # Reshape shape (batch, hidden)
            # return tensor[:, -1, :] for tensors without sequence
            # for tensors with sequence, return
            return tensor

        # Output shape (features, hidden)
        return x[0]

class CaptioningModel(Module):
    def __init__(self, emb_dim, emb_mat, vocab_size, input_shape, start_token_idx, end_token_idx, truncation_length, *args, **kwargs):
        super(CaptioningModel, self).__init__(*args, **kwargs)
        self.device = None
        self.init_params = (emb_dim, emb_mat, vocab_size, input_shape, start_token_idx, end_token_idx, truncation_length, args, kwargs)
        init_param_names = ('emb_dim', 'emb_mat', 'vocab_size', 'input_shape', 'start_token_idx', 'end_token_idx', 'truncation_length', 'args', 'kwargs')
        self.init_param_dict = dict(zip(init_param_names, self.init_params))

        # Image layers
        self.CNN, cnn_out_shape = CaptioningModel.CNN_Block(input_shape)

        cnn_shape = (input_shape, cnn_out_shape)

        # Embedding layer
        vocab_size = emb_mat.shape[0]
        self.emb_layer = Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=0)
        self.emb_layer.weight = torch.nn.Parameter(torch.tensor(emb_mat, dtype=torch.float32))
        self.emb_layer.weight.requires_grad = False

        # LSTM layers
        self.LSTM_Encoder, lstm_out_shape = CaptioningModel.LSTM_Block(cnn_shape, emb_dim, dropout=0.2)

        # Concatenate layer
        self.output = Sequential(OrderedDict([
            ('logits', Linear(lstm_out_shape[0], emb_mat.shape[0])),
            # Assume logits as (batch_size, vocab_size)
            ('softmax', LogSoftmax(dim=1))
        ]))

    def remove_softmax(self):
        output_modules = list(self.output.named_children())
        softmax_idx = -1
        for idx, (name, module) in reversed(list(enumerate(output_modules))):
            if name == 'softmax':
                softmax_idx = idx
                break

        if softmax_idx == -1:
            raise ValueError("Softmax layer not found!")

        self.output.pop(softmax_idx)

    @staticmethod
    def CNN_Block(input_shape):
        C, H, W = input_shape

        conv_layers = Sequential(
            Conv2d(in_channels=C, out_channels=32, kernel_size=(3, 3), padding='same'),
            BatchNorm2d(32),
            MaxPool2d(kernel_size=2, stride=2),
            ReLU(),
            Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding='same'),
            MaxPool2d(kernel_size=2, stride=2),
            ReLU(),
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding='same'),
            MaxPool2d(kernel_size=2, stride=2),
            ReLU(),
            Flatten(),
        )

        # Calculate the output shape
        def dummy_forward(conv_layers, C, H, W) -> torch.Tensor:
            with torch.no_grad():
                x = torch.randn((1, C, H, W))
                x = conv_layers(x)
            no_batch = x.squeeze(0)
            if not x.dim() > no_batch.dim():
                raise ValueError("First batch dimension must be preserved!")
            x = no_batch
            return x

        x = dummy_forward(conv_layers, C, H, W)
        conv_layers.append(Linear(x.numel(), 256))
        x = dummy_forward(conv_layers, C, H, W)  # Should be of shape (256,)

        return conv_layers, x.shape

    @staticmethod
    def LSTM_Block(cnn_shape: Tuple[Tuple, Tuple], emb_dim, dropout=0.2):
        cnn_in, cnn_out = cnn_shape

        if len(cnn_out) != 1:
            raise ValueError("Expected a single output shape from the CNN block!")

        concat_in = cnn_out[0] + emb_dim  # It's expected that the image features are concatenated with the embeddings
        lstm_layers = Sequential(
            OrderedDict([
                ('lstm1', LSTM(concat_in, emb_dim, num_layers=2, batch_first=True, dropout=dropout)),
                ('_lstm1_extract', extract_tensor()),
                ('dropout', Dropout(dropout)),  # Add last dropout layer
                ('act1', LeakyReLU(True)),
                ('lstm2', LSTM(emb_dim, emb_dim, num_layers=2, batch_first=True, dropout=dropout)),
                ('_lstm2_extract', extract_tensor()),
                ('dropout2', Dropout(dropout)),  # Add last dropout layer
                ('act2', LeakyReLU(True))
            ])
        )

        return lstm_layers, (emb_dim,)

    def forward(self, image, captions=None, mode='train'):
        """
        Forward pass for the CaptioningModel.

        Args:
        image (Tensor): The input image tensor.
        captions (Tensor): The ground truth captions for training. Not used during inference.
        mode (str): Mode of operation, 'train' or 'infer'.

        Returns:
        Tensor: The output from the captioning model.
        """
        # Call flatten_parameters() to avoid errors with LSTM
        for m in self.LSTM_Encoder.children():
            if isinstance(m, LSTM):
                m.flatten_parameters()

        # Process image through CNN
        image_features = self.CNN(image)

        if mode == 'train':
            if captions is None:
                # Fill with 0s if no captions are provided
                captions = torch.zeros((image.size(0), self.init_param_dict['truncation_length']), dtype=torch.long, device=image.device)
                # Set first to start token
                captions[:, 0] = self.init_param_dict['start_token_idx']

            # Concatenate image features with captions for each timestep
            captions = self.emb_layer(captions)
            inputs = torch.cat([torch.stack([image_features]*captions.size(1), dim=1), captions], dim=2)
            outputs = self.LSTM_Encoder(inputs)
            return self.output(outputs)  # This assumes processing of all time-steps together

        elif mode == 'infer':
            # Start with an initial token (e.g., '<start>') and generate captions
            generated = []

            start_token_class_idx = self.init_param_dict['start_token_idx']
            # Create a <start> token tensor with the same batch size as the image
            caption = torch.tensor([start_token_class_idx]*image.size(0), dtype=torch.long, device=image.device).unsqueeze(1)
            caption = self.emb_layer(caption)

            # Concatenate image features with the <start> token

            image_features = torch.stack([image_features]*caption.size(1), dim=1)
            current_input = torch.cat([image_features, caption], dim=2)

            for _ in range(self.init_param_dict['truncation_length']):
                lstm_out = self.LSTM_Encoder(current_input)
                output = self.output(lstm_out)
                _, predicted = torch.max(output, 2)
                generated.append(predicted)

                predicted = self.emb_layer(predicted)
                current_input = torch.cat([image_features, predicted], dim=2)

            return torch.stack(generated, dim=1)

    def to_device(self, device):
        self.device = device
        self.to(device)
        for m in self.LSTM_Encoder.children():
            if isinstance(m, LSTM):
                m.flatten_parameters()



# Multimodal model that combines image and text branches using Concatenate
def multimodal_model(max_length, vocab_size, image_shape, emb_dim, emb_mat, end_token_idx, truncation_length):
    # Define image model
    model = CaptioningModel(emb_dim, emb_mat, vocab_size, image_shape, 1, 15, truncation_length)
    return model
