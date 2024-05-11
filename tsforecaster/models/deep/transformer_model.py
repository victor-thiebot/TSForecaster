import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads, hidden_dim, output_dim):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
        )
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        output = self.transformer(x, x)
        output = self.fc(output)
        return output
