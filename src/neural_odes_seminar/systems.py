"""Known differential equations and their analytical reference solutions."""

import torch


def decay_solution(t: torch.Tensor, rate: float = 0.8, y0: float = 1.0) -> torch.Tensor:
    """Exact solution of y' = -rate * y."""
    return y0 * torch.exp(-rate * t)


def decay_derivative(y: torch.Tensor, rate: float = 0.8) -> torch.Tensor:
    return -rate * y


def si_solution(t: torch.Tensor, beta: float = 1.0, infected0: float = 0.01) -> torch.Tensor:
    """Exact normalized SI solution, returned as columns (S, I)."""
    susceptible0 = 1.0 - infected0
    infected = 1.0 / (1.0 + (susceptible0 / infected0) * torch.exp(-beta * t))
    susceptible = 1.0 - infected
    return torch.stack((susceptible, infected), dim=-1)


def si_derivative(state: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    susceptible, infected = state[..., 0], state[..., 1]
    flow = beta * susceptible * infected
    return torch.stack((-flow, flow), dim=-1)
