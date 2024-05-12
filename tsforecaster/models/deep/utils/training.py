import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def get_optimizer(model, optimizer_name, lr):
    """
    Initializes and returns the specified optimizer for the given model.
    """
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")


def get_loss_fn(loss_fn_name):
    """
    Returns the specified loss function.
    """
    if loss_fn_name == "mse":
        return nn.MSELoss()
    elif loss_fn_name == "mae":
        return nn.L1Loss()
    else:
        raise ValueError(f"Invalid loss function name: {loss_fn_name}")


def get_device():
    """
    Returns the device (CPU or GPU) to be used for training.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, data_module, epochs, optimizer, loss_fn, device, eval_frequency=1):
    """
    Trains the given model using the specified data module, optimizer, loss function, and device.
    Returns the training and validation losses.
    """
    model.to(device)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_loss = train_epoch(
            model, data_module.train_dataloader(), optimizer, loss_fn, device
        )
        train_losses.append(train_loss)

        if (epoch + 1) % eval_frequency == 0:
            val_loss = eval_epoch(model, data_module.val_dataloader(), loss_fn, device)
            val_losses.append(val_loss)
            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
        else:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

    return train_losses, val_losses


def train_epoch(model, data_loader, optimizer, loss_fn, device):
    """
    Performs one epoch of training for the given model.
    """
    model.train()
    epoch_loss = 0.0
    for data in data_loader:
        x, y = data[0], data[1]  # Assuming input and target are the first two elements
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)


def eval_epoch(model, data_loader, loss_fn, device):
    """
    Performs one epoch of evaluation for the given model.
    """
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for data in data_loader:
            x, y = (
                data[0],
                data[1],
            )  # Assuming input and target are the first two elements
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)
