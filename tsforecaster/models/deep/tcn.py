import torch
import torch.nn as nn
from .base_model import Model


class TCN(Model):
    def __init__(self, input_dim, output_dim, hidden_sizes, kernel_sizes, dropout=0.1):
        super().__init__(input_dim, output_dim)
        self.hidden_sizes = hidden_sizes
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout

        self.tcn_layers = nn.ModuleList()
        in_channels = input_dim
        for hidden_size, kernel_size in zip(hidden_sizes, kernel_sizes):
            self.tcn_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            in_channels = hidden_size

        self.linear = nn.Linear(hidden_sizes[-1], output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_length)

        for layer in self.tcn_layers:
            x = nn.functional.relu(layer(x))
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        x = x.permute(0, 2, 1)  # (batch_size, seq_length, hidden_size)
        x = self.linear(x[:, -1, :])  # (batch_size, output_dim)

        return x


# example use
# model = TCN(
#     input_dim=10,
#     output_dim=5,
#     hidden_sizes=[64, 128, 256],
#     kernel_sizes=[3, 5, 7],
#     dropout=0.2
# )
