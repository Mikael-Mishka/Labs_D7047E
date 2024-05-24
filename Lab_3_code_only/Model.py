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
            tensor, (hidden_state, cell_state) = x
            # Reshape shape (batch, hidden)
            # return tensor[:, -1, :] for tensors without sequence
            # for tensors with sequence, return
            return tensor

        # Output shape (features, hidden)
        return x[0]


class extract_tensor_bi(Module):
    def forward(self, x):
        # For bidirectional LSTM output handling
        if len(x[0].shape) > 2:
            tensor, (hidden_state, cell_state) = x
            # Concatenate the outputs for both directions
            seq_len = tensor.shape[2] // 2
            forward = tensor[:, :, :seq_len]  # last time step, forward direction
            backward = tensor[:, :, seq_len:]  # first time step, backward direction
            return torch.cat((forward, backward), dim=1)  # concatenate along feature dimension

        # For single layer outputs (non-sequence)
        return x[0]

class CaptioningModel(Module):
    def __init__(self, emb_dim, emb_mat, vocab_size, input_shape, start_token_idx, end_token_idx, truncation_length, *args, **kwargs):
        super(CaptioningModel, self).__init__(*args, **kwargs)
        self.device = None
        self.init_params = (emb_dim, emb_mat, vocab_size, input_shape, start_token_idx, end_token_idx, truncation_length, args, kwargs)
        init_param_names = ('emb_dim', 'emb_mat', 'vocab_size', 'input_shape', 'start_token_idx', 'end_token_idx', 'truncation_length', 'args', 'kwargs')
        self.init_param_dict = dict(zip(init_param_names, self.init_params))

        # Image layers
        self.CNN, cnn_out_shape = CaptioningModel.CNN_Block(input_shape, emb_dim)

        cnn_shape = (input_shape, cnn_out_shape)

        # Embedding layer
        vocab_size = emb_mat.shape[0]
        self.emb_layer = Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=0, scale_grad_by_freq=True)
        self.emb_layer.weight = torch.nn.Parameter(torch.tensor(emb_mat, dtype=torch.float32))
        self.emb_layer.weight.requires_grad = True
        self.emb_layer = Sequential(OrderedDict([
            ('embedding', self.emb_layer),
            ('layer_norm', torch.nn.LayerNorm([emb_dim])),
        ]))


        # LSTM layers
        self.LSTM_Encoder, lstm_out_shape = CaptioningModel.LSTM_Block(cnn_shape, emb_dim, vocab_size, dropout=0.2)

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
    def CNN_Block(input_shape: Tuple, emb_dim):
        C, H, W = input_shape

        # TESTING
        # conv_layers = Sequential(
        #     Conv2d(in_channels=C, out_channels=32, kernel_size=(3, 3), padding='same'),
        #     BatchNorm2d(32),
        #     MaxPool2d(kernel_size=2, stride=2),
        #     ReLU(),
        #     Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding='same'),
        #     MaxPool2d(kernel_size=2, stride=2),
        #     ReLU(),
        #     Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding='same'),
        #     MaxPool2d(kernel_size=2, stride=2),
        #     ReLU(),
        #     Flatten(),
        # )

        # # Calculate the output shape
        # def dummy_forward(conv_layers, C, H, W) -> torch.Tensor:
        #     with torch.no_grad():
        #         x = torch.randn((1, C, H, W))
        #         x = conv_layers(x)
        #     no_batch = x.squeeze(0)
        #     if not x.dim() > no_batch.dim():
        #         raise ValueError("First batch dimension must be preserved!")
        #     x = no_batch
        #     return x
#
        # x = dummy_forward(conv_layers, C, H, W)
        # print(x.numel())
        # conv_layers.append(Linear(x.numel(), 256))
        # x = dummy_forward(conv_layers, C, H, W)  # Should be of shape (256,)

        # Use VGG16 architecture
        from torchvision.models import vgg16
        from torchvision.models.vgg import VGG16_Weights
        vgg16_model = vgg16(weights=VGG16_Weights.DEFAULT)
        # Remove vgg16 classifier layers
        feature_out = vgg16_model.classifier[-1].in_features
        vgg16_model.classifier = vgg16_model.classifier[:-1]
        # Set requires_grad to False on original model
        for param in vgg16_model.parameters():
            param.requires_grad = False

        head = Sequential(
            Linear(feature_out, emb_dim),
            torch.nn.LayerNorm([emb_dim]),
            LeakyReLU(True)
        )
        vgg16_model.classifier.extend(head)
        head.requires_grad_(True)

        conv_layers = vgg16_model
        x = torch.randn((1, C, H, W))
        x = conv_layers(x)
        x = x.squeeze(0)

        if C != 3:
            raise ValueError("Expected 3 channels for input image!")

        return conv_layers, x.shape

    @staticmethod
    def LSTM_Block(cnn_shape: Tuple[Tuple, Tuple], emb_dim, vocab_size, dropout=0.2):
        cnn_in, cnn_out = cnn_shape

        if len(cnn_out) != 1:
            raise ValueError("Expected a single output shape from the CNN block!")

        concat_in = cnn_out[0] + 1
        lstm_layers = Sequential(
            OrderedDict([
                ('lstm1', LSTM(concat_in, emb_dim, num_layers=2, batch_first=True, bidirectional=False)),
                ('_lstm1_extract', extract_tensor()),
                ('dropout1', Dropout(dropout)),  # Add last dropout layer
                ('act1', LeakyReLU(True)),

                ('lstm2', LSTM(emb_dim, emb_dim, num_layers=2, batch_first=True, bidirectional=False)),
                ('_lstm2_extract', extract_tensor()),
                ('dropout2', Dropout(dropout)),  # Add last dropout layer
                ('act2', LeakyReLU(True)),

                ('dense', Linear(emb_dim, vocab_size // 8)),
                ('dropout3', Dropout(dropout)),
                ('act3', LeakyReLU(True)),
            ])
        )

        return lstm_layers, (lstm_layers.dense.out_features,)

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
            image_features = torch.stack([image_features]*captions.size(1), dim=1)
            # inputs = torch.cat([torch.stack([image_features]*captions.size(1), dim=1), captions], dim=2)
            # captions = captions / torch.norm(captions, dim=2, keepdim=True)
            # Replace nans with 0s
            # captions[torch.isnan(captions)] = 0
            # image_features = image_features / torch.norm(image_features, dim=2, keepdim=True)

            inputs = torch.bmm(captions.view(-1, 1, captions.size(2)), image_features.view(-1, image_features.size(2), 1))
            # inputs[torch.isnan(inputs)] = 0
            inputs = inputs.squeeze(2).view(-1, captions.size(1), 1)
            inputs = torch.cat([inputs, image_features], dim=2)
            outputs = self.LSTM_Encoder(inputs)
            return self.output(outputs)  # This assumes processing of all time-steps together
        elif mode == 'infer':
            # Start with an initial token (e.g., '<start>') and generate captions
            ongoing = torch.zeros((image.size(0), self.init_param_dict['truncation_length']), dtype=torch.long, device=image.device)
            generated = torch.zeros((image.size(0), self.init_param_dict['truncation_length']), dtype=torch.long, device=image.device)

            start_token_class_idx = self.init_param_dict['start_token_idx']

            # Create a <start> token tensor with the same batch size as the image

            caption = torch.tensor([start_token_class_idx] * image.size(0), dtype=torch.long, device=image.device).unsqueeze(1)
            caption = self.emb_layer(caption)

            image_features = torch.stack([image_features] * caption.size(1), dim=1)

            # current_input = torch.cat([image_features, caption], dim=2)
            # caption = caption / torch.norm(caption, dim=2, keepdim=True)
            # Sometimes embeddings of captions are all 0s, normalization does not play nice with 0s
            # captions[torch.isnan(captions)] = 0
            # image_features = image_features / torch.norm(image_features, dim=2, keepdim=True)
            current_input = torch.bmm(caption.view(-1, 1, caption.size(2)), image_features.view(-1, image_features.size(2), 1))
            current_input[torch.isnan(current_input)] = 0
            current_input = current_input.squeeze(2).view(-1, caption.size(1), 1)

            hidden_states = None  # Initialize hidden states for LSTM layers
            hidden_states2 = None  # Initialize hidden states for LSTM layers

            completed_batch = torch.tensor([], device= image.device) # List of batches that have completed generating captions
            for _ in range(self.init_param_dict['truncation_length']):
                current_input = torch.cat([current_input, image_features], dim=2)
                lstm_out, hidden_states = self.LSTM_Encoder.lstm1(current_input, hidden_states)
                current_input = self.LSTM_Encoder._lstm1_extract((lstm_out, hidden_states))
                current_input = self.LSTM_Encoder.dropout1(current_input)
                current_input = self.LSTM_Encoder.act1(current_input)

                lstm_out, hidden_states2 = self.LSTM_Encoder.lstm2(current_input, hidden_states2)
                current_input = self.LSTM_Encoder._lstm2_extract((lstm_out, hidden_states2))
                current_input = self.LSTM_Encoder.dropout2(current_input)
                current_input = self.LSTM_Encoder.act2(current_input)

                current_input = self.LSTM_Encoder.dense(current_input)
                current_input = self.LSTM_Encoder.dropout3(current_input)
                current_input = self.LSTM_Encoder.act3(current_input)
                output = self.output(current_input)

                # argmax approach
                # predicted = torch.argmax(output, 2)

                # topk multinomial approach
                k = 3

                output_topk = torch.topk(output.view(-1, output.size(2)), k, dim=1)
                predicted = torch.multinomial(F.softmax(output_topk.values, dim=1), 1)
                predicted = output_topk.indices.gather(1, predicted)

                ongoing[:, _] = predicted.squeeze(1)

                # Extract batches that have completed generating captions
                # shape: (N,)
                predicted_temp = predicted.clone()
                for idx in completed_batch:
                    idx = int(idx)
                    # Insert indices so that the completed batches indices are relative to the first dimension
                    zero_tensor = torch.zeros((1, predicted_temp.size(1)), dtype=torch.long, device=image.device)
                    predicted_temp = torch.cat([predicted_temp[:idx], zero_tensor, predicted_temp[idx:]], dim=0)


                completed_indices_shift = torch.nonzero(predicted == self.init_param_dict['end_token_idx'], as_tuple=True)[0]
                completed_indices = torch.nonzero(predicted_temp == self.init_param_dict['end_token_idx'], as_tuple=True)[0]
                # Sort the completed batches in descending order
                completed_indices = torch.sort(completed_indices, descending=True)[0]
                completed_indices_shift = torch.sort(completed_indices_shift, descending=True)[0]
                if len(completed_indices) > 0:
                    # Remove completed batches from the current input
                    ongoing_temp = ongoing.clone()
                    # Reinstate the completed batches in ongoing_temp to avoid indexing issues with generated
                    for idx in completed_batch:
                        idx = int(idx)
                        ongoing_temp = torch.cat([ongoing_temp[:idx,:], generated[idx,:].unsqueeze(0), ongoing_temp[idx:,:]], dim=0)
                    completed_batch = torch.cat([completed_batch, completed_indices], dim=0)
                    # Sort the completed batches in ascending order
                    completed_batch = torch.sort(completed_batch)[0]

                    for idx in completed_indices:
                        generated[idx, :] = ongoing_temp[idx, :]

                    for idx in completed_indices_shift:
                        image_features = torch.cat([image_features[:idx], image_features[idx + 1:]], dim=0)
                        hidden_states =  torch.cat([hidden_states[0][:, :idx], hidden_states[0][:, idx+1:]], dim=1), \
                                         torch.cat([hidden_states[1][:, :idx], hidden_states[1][:, idx+1:]], dim=1)
                        hidden_states2 = torch.cat([hidden_states2[0][:, :idx], hidden_states2[0][:, idx+1:]], dim=1), \
                                         torch.cat([hidden_states2[1][:, :idx], hidden_states2[1][:, idx+1:]], dim=1)
                        predicted = torch.cat([predicted[:idx], predicted[idx + 1:]], dim=0)
                        ongoing = torch.cat([ongoing[:idx], ongoing[idx + 1:]], dim=0)

                    if ongoing.size(0) == 0:
                        break

                predicted = self.emb_layer(predicted)
                # current_input = torch.cat([image_features, predicted], dim=2)
                # current_input = torch.add(image_features, predicted)
                # predicted = predicted / torch.norm(predicted, dim=2, keepdim=True)
                # predicted[torch.isnan(predicted)] = 0
                # image_features = image_features / torch.norm(image_features, dim=2, keepdim=True)
                current_input = torch.bmm(predicted.view(-1, 1, predicted.size(2)), image_features.view(-1, image_features.size(2), 1))
                current_input = current_input.squeeze(2).view(-1, predicted.size(1), 1)

            return generated

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
