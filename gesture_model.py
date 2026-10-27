# Small GRU sequence classifier for the dynamic gestures.

import torch
from torch import nn


class GestureGRU(nn.Module):
    def __init__(self, n_features, n_classes, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, n_classes))

    def forward(self, x):  # x: (batch, time, features)
        _, h = self.gru(x)
        return self.head(h[-1])


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = GestureGRU(ckpt["n_features"], len(ckpt["labels"]), ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
