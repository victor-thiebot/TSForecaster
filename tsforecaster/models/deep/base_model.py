import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, model_type, model_params):
        super(Model, self).__init__()
        self.model_type = model_type
        self.model = self.build_model(model_params)

    def build_model(self, model_params):
        if self.model_type == "lstm":
            return LSTMModel(**model_params)
        elif self.model_type == "gru":
            return GRUModel(**model_params)
        elif self.model_type == "transformer":
            return TransformerModel(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def forward(self, x):
        return self.model(x)
