import torch
import torch.nn as nn


class Model(nn.Module):
    # def __init__(self, input_dim, output_dim):
    def __init__(self):
        super().__init__()
        # self.input_dim = input_dim
        # self.output_dim = output_dim
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this!")

    def fit(
        self,
        train_loader,
        val_loader,
        epochs,
        optimizer,
        criterion,
        device,
        eval_frequency=1,
    ):
        self.to(device)
        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, optimizer, criterion, device)
            self.train_losses.append(train_loss)

            if (epoch + 1) % eval_frequency == 0:
                val_loss = self._val_epoch(val_loader, criterion, device)
                self.val_losses.append(val_loss)
                print(
                    f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

    def _train_epoch(self, train_loader, optimizer, criterion, device):
        self.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = self(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(train_loader)

    def _val_epoch(self, val_loader, criterion, device):
        self.eval()
        epoch_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_hat = self(x)
                loss = criterion(y_hat, y)
                epoch_loss += loss.item()
        return epoch_loss / len(val_loader)
