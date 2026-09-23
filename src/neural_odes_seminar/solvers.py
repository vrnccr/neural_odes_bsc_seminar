"""A deliberately small differentiable forward Euler solver."""

import torch


def euler_solve(derivative, initial_state: torch.Tensor, observation_times: torch.Tensor,
                step_size: float = 0.01) -> torch.Tensor:
    """Solve an autonomous ODE and return states at requested equally spaced times.

    The function stays in PyTorch operations, so autograd can differentiate the
    loss through every Euler step. All observation times must begin with zero.
    """
    if observation_times[0].item() != 0:
        raise ValueError("observation_times must begin at 0")
    increments = observation_times[1:] - observation_times[:-1]
    state = initial_state
    states = [state]
    for interval in increments:
        steps = round(float(interval.item()) / step_size)
        if not torch.isclose(interval, torch.tensor(steps * step_size, dtype=interval.dtype)):
            raise ValueError("each observation interval must be a multiple of step_size")
        for _ in range(steps):
            state = state + step_size * derivative(state)
        states.append(state)
    return torch.stack(states, dim=0)
