"""Small neural networks used as learned rules of change."""

import torch
from torch import nn


class SmallMLP(nn.Module):
    """One hidden tanh layer and a linear output layer."""

    def __init__(self, inputs: int, outputs: int, hidden: int = 16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, outputs)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)
