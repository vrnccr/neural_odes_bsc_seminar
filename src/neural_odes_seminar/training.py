"""Training functions shared by notebooks and the command-line runner."""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .solvers import euler_solve


@dataclass
class TrainingResult:
    history: list[float]
    predictions: torch.Tensor


def fit_trajectory(derivative: Callable, parameters, initial_state: torch.Tensor,
                   observation_times: torch.Tensor, observations: torch.Tensor,
                   steps: int = 2000, learning_rate: float = 0.001,
                   euler_step: float = 0.01) -> TrainingResult:
    """Fit an ODE rule by minimizing trajectory mean squared error."""
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    history = []
    for _ in range(steps):
        optimizer.zero_grad()
        predicted = euler_solve(derivative, initial_state, observation_times, euler_step)
        loss = torch.mean((predicted - observations) ** 2)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    with torch.no_grad():
        predictions = euler_solve(derivative, initial_state, observation_times, euler_step)
    return TrainingResult(history, predictions)
