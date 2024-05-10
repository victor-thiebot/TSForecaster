import os
import torch

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_state = torch.load(model_path)
    model_type = model_state['model_type']
    model_params = model_state['model_params']

    if model_type == 'LSTM':
        from .lstm_model import LSTMModel
        model = LSTMModel(**model_params)
    elif model_type == 'GRU':
        from .gru_model import GRUModel
        model = GRUModel(**model_params)
    elif model_type == 'Transformer':
        from .transformer_model import TransformerModel
        model = TransformerModel(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.load_state_dict(model_state['model_state_dict'])
    model.eval()
    return model