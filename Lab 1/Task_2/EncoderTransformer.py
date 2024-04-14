import torch.nn as nn
from torch import Tensor
import math
import torch


class PositionalEncoder(nn.Module):
    def __init__(self, max_len, d_model, device):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.dropout = nn.Dropout(0.1)

        pe = torch.zeros((max_len, d_model))
        pe = pe.to(self.device)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # Input tensor to Postional encoder forward method: [max_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Transformer(nn.Module):

    def __init__(self, vocab_size, ntokens, d_model, nhead, num_encoder_layers,dim_feedforward, dropout, device, BATCH_SIZE):
        super().__init__()

        self.model_type = 'Transformer'

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.ntokens = ntokens # 484
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Embedding(vocab_size, d_model).to(self.device)

        self.positional_encoder = PositionalEncoder(ntokens, d_model, self.device).to(self.device)

        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model,
                                                                        nhead=nhead,
                                                                        dim_feedforward=dim_feedforward,
                                                                        dropout=0.1,
                                                                        device=self.device, batch_first=True), num_layers=num_encoder_layers).to(self.device)

        self.BATCH_SIZE = BATCH_SIZE

        print(self.BATCH_SIZE*self.d_model)

        self.flatten = nn.Flatten().to(self.device)

        self.linear = nn.Linear(ntokens * d_model, 5).to(self.device)


    # Forward pass to classify amazon review as either 1, 2, 3, 4, 5 stars
    def forward(self, src: Tensor) -> Tensor:

        vocab_size = src.shape[0]
        src = src.to(device=self.device)

        embedded = self.embedding(src)
        src = embedded * math.sqrt(self.d_model)

        #print(f"Before positional encoder: {src.shape}")

        positional_encoder_output = self.positional_encoder(src)


        # After positional encoder:
        #print(f"After positional encoder: {positional_encoder_output.shape}")

        output = self.encoder(positional_encoder_output)

        #print(f"Output shape after encoder: {output.shape}")


        output = self.flatten(output)

        #print(f"Output shape after view: {output.shape}")

        output = self.linear(output)
        return output