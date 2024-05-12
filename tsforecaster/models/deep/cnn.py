import torch
import torch.nn as nn
from .base_model import Model


class CNN(Model):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_filters=64,
        kernel_size=3,
        stride=1,
        dilation=1,
        padding=0,
        activation="ReLU",
    ):
        super().__init__(input_dim, output_dim)
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)  # Add adaptive average pooling
        self.activation = getattr(nn, activation)()
        self.linear = nn.Linear(num_filters, output_dim)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.activation(x)
        x = self.adaptive_pool(x)  # Apply adaptive average pooling
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.linear(x)
        return x
