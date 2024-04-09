from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder


# This class will be the main trainformer model
class Transformer(nn.Module):

    def __init__(self, n_tokens, d_model, nhead, d_dim, n_layers, dropout=0.3):
        super().__init__()

        # Define the positional encoder to keep track of the position of the words
        self.positional_encoder = PositionalEncoder()

        # Define the encoder layers
        self.encoder_layer = TransformerEncoderLayer(d_model, nhead, d_dim, dropout)

        # Define the encoder
        self.encoder = TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=n_layers)

        # Define the embedding layer
        self.embedding = nn.Embedding(n_tokens, d_model)


    def forward(self, x):
        pass


# This will be the class that implements the positional encoder
class PositionalEncoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass