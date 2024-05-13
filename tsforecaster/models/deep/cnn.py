import torch
import torch.nn as nn
from .base_model import Model


## THIS 0NE WORKS FINE --> DO NOT CHANGE
class CNN(Model):
    def __init__(
        self,
        n_channels_x,
        n_channels_y,
        window_length_x,
        window_length_y,
        num_filters=64,
        kernel_size=3,
        stride=1,
        dilation=1,
        padding=0,
        activation="ReLU",
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=n_channels_x,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.activation = getattr(nn, activation)()
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(
            num_filters * (window_length_x - 2 * (kernel_size - 1)),
            n_channels_y * window_length_y,
        )
        self.n_channels_y = n_channels_y
        self.window_length_y = window_length_y

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = x.reshape(x.size(0), self.n_channels_y, self.window_length_y)
        return x


# class CNN(Model):
#     def __init__(
#         self,
#         n_channels_x,
#         n_channels_y,
#         window_length_x,
#         window_length_y,
#         conv_layers=None,
#         fc_layers=None,
#         activation="ReLU",
#     ):
#         super().__init__()
#         self.conv_layers = nn.ModuleList(conv_layers or [])
#         self.fc_layers = nn.ModuleList(fc_layers or [])
#         self.activation = getattr(nn, activation)()
#         self.n_channels_y = n_channels_y
#         self.window_length_y = window_length_y

#         # Calculate the output size of the convolutional layers
#         conv_output_size = window_length_x
#         for layer in self.conv_layers:
#             conv_output_size = (
#                 conv_output_size - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1
#             ) // layer.stride[0] + 1

#         # Add a flatten layer
#         self.flatten = nn.Flatten()

#         # Calculate the input size for the first fully connected layer
#         fc_input_size = self.conv_layers[-1].out_channels * conv_output_size

#         # Add the fully connected layers
#         for i in range(len(self.fc_layers)):
#             if i == 0:
#                 self.fc_layers[i] = nn.Linear(
#                     fc_input_size, self.fc_layers[i].out_features
#                 )
#             else:
#                 self.fc_layers[i] = nn.Linear(
#                     self.fc_layers[i - 1].out_features, self.fc_layers[i].out_features
#                 )

#         # Add the final linear layer to map to the desired output shape
#         self.fc_layers.append(
#             nn.Linear(self.fc_layers[-1].out_features, n_channels_y * window_length_y)
#         )

#     def forward(self, x):
#         for layer in self.conv_layers:
#             x = layer(x)
#             x = self.activation(x)
#         x = self.flatten(x)
#         for layer in self.fc_layers[:-1]:
#             x = layer(x)
#             x = self.activation(x)
#         x = self.fc_layers[-1](x)
#         x = x.reshape(x.size(0), self.n_channels_y, self.window_length_y)
#         return x
