import torch


def one_euler_step_loss(theta: torch.Tensor) -> torch.Tensor:
    """Loss after y_1 = y_0 - h theta y_0, with y_0 = 1 and h = 0.1."""
    y1 = 1.0 - 0.1 * theta
    return (y1 - 0.9) ** 2


def test_autograd_matches_finite_difference_for_one_euler_step():
    theta = torch.tensor(0.3, requires_grad=True)
    one_euler_step_loss(theta).backward()
    automatic = theta.grad.item()
    epsilon = 1e-4
    finite_difference = (
        one_euler_step_loss(torch.tensor(0.3 + epsilon))
        - one_euler_step_loss(torch.tensor(0.3 - epsilon))
    ).item() / (2 * epsilon)
    assert abs(automatic - finite_difference) < 1e-3
