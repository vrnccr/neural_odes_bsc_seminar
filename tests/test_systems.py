import torch

from neural_odes_seminar.solvers import euler_solve
from neural_odes_seminar.systems import decay_derivative, decay_solution, si_solution


def test_decay_exact_solution_satisfies_initial_value():
    assert torch.isclose(decay_solution(torch.tensor(0.0)), torch.tensor(1.0))


def test_si_solution_conserves_population():
    states = si_solution(torch.linspace(0, 15, 20), infected0=0.1)
    assert torch.allclose(states.sum(dim=-1), torch.ones(20))


def test_euler_improves_when_step_size_is_smaller():
    initial = torch.tensor([1.0])
    end = torch.tensor([0.0, 5.0])
    coarse = euler_solve(decay_derivative, initial, end, 0.5)[-1]
    fine = euler_solve(decay_derivative, initial, end, 0.05)[-1]
    exact = decay_solution(torch.tensor(5.0))
    assert torch.abs(fine - exact) < torch.abs(coarse - exact)
