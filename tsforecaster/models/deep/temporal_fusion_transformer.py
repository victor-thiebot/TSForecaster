import torch
import torch.nn as nn
from .base_model import Model


class TemporalFusionTransformer(Model):
    def __init__(
        self, input_dim, output_dim, hidden_size, num_heads, num_layers, dropout=0.1
    ):
        super().__init__(input_dim, output_dim)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Linear(input_dim, hidden_size)
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size, nhead=num_heads, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        batch_size, seq_length, _ = x.size()

        # Embedding layer
        x = self.embedding(x)  # (batch_size, seq_length, hidden_size)

        # Positional encoding
        positions = torch.arange(
            seq_length, dtype=torch.long, device=x.device
        ).unsqueeze(0)
        x += self.positional_encoding(positions)

        # Dropout
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x)

        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, hidden_size)

        # Decoder
        x = self.decoder(x)  # (batch_size, output_dim)

        return x

    def positional_encoding(self, positions):
        d_model = self.hidden_size
        angles = positions / torch.pow(
            10000, (2 * (torch.arange(d_model) // 2)) / d_model
        )
        angles[:, 0::2] = torch.sin(angles[:, 0::2])
        angles[:, 1::2] = torch.cos(angles[:, 1::2])
        return angles


## example use
# model = TemporalFusionTransformer(
#     input_dim=10,
#     output_dim=5,
#     hidden_size=128,
#     num_heads=4,
#     num_layers=2,
#     dropout=0.2
# )
