"""Run every teaching experiment and write report-ready figures and results.

Run from the repository root:
    python -m neural_odes_seminar.run
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# The examples use many very small tensor operations. One CPU thread avoids
# parallelisation overhead and makes the notebooks responsive on ordinary laptops.
torch.set_num_threads(1)

from .models import SmallMLP
from .plotting import setup
from .solvers import euler_solve
from .systems import decay_derivative, decay_solution, si_derivative, si_solution
from .training import fit_trajectory


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "report" / "figures"
METRICS = ROOT / "results" / "metrics"
HISTORIES = ROOT / "results" / "histories"
SEED = 7
EULER_STEP = 0.01
SI_EULER_STEP = 0.2


def save_history(name: str, values: list[float]) -> None:
    with (HISTORIES / f"{name}.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["update", "loss"])
        writer.writerows(enumerate(values, start=1))


def save_metrics(metrics: dict) -> None:
    (METRICS / "experiment_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


def experiment_1() -> dict:
    times = torch.linspace(0, 5, 101)
    exact = decay_solution(times)
    initial = torch.tensor([1.0])
    coarse_times = torch.linspace(0, 5, 11)
    coarse = euler_solve(lambda y: decay_derivative(y), initial, coarse_times, 0.5).squeeze()
    fine = euler_solve(lambda y: decay_derivative(y), initial, times, 0.05).squeeze()
    fig, axis = plt.subplots(figsize=(6.2, 3.7))
    axis.plot(times, exact, color="#202020", label="exact solution")
    axis.plot(coarse_times, coarse, "o--", color="#c84b31", label="Euler, h = 0.5")
    axis.plot(times, fine, color="#2f6f9f", label="Euler, h = 0.05")
    axis.set(xlabel="time t", ylabel="amount y(t)", ylim=(0, 1.05))
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "01_euler_decay.pdf")
    plt.close(fig)
    return {
        "maximum_error_h_0_5": float(torch.max(torch.abs(coarse - decay_solution(coarse_times)))),
        "maximum_error_h_0_05": float(torch.max(torch.abs(fine - exact))),
    }


def experiment_2() -> dict:
    times = torch.arange(0, 5.01, 0.5)
    observations = decay_solution(times).unsqueeze(-1)
    initial = torch.tensor([1.0])
    theta = torch.nn.Parameter(torch.tensor(0.3))
    theta_history = []

    def derivative(y):
        return -theta * y

    optimizer = torch.optim.Adam([theta], lr=0.001)
    loss_history = []
    initial_prediction = euler_solve(derivative, initial, times, EULER_STEP).detach().squeeze()
    for _ in range(2000):
        optimizer.zero_grad()
        predicted = euler_solve(derivative, initial, times, EULER_STEP)
        loss = torch.mean((predicted - observations) ** 2)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach()))
        theta_history.append(float(theta.detach()))
    final_prediction = euler_solve(derivative, initial, times, EULER_STEP).detach().squeeze()
    save_history("02_decay_parameter_loss", loss_history)
    with (HISTORIES / "02_decay_parameter_theta.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["update", "theta"])
        writer.writerows(enumerate(theta_history, start=1))

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    axes[0].plot(times, observations.squeeze(), "ko", ms=3, label="observations")
    axes[0].plot(times, initial_prediction, "--", color="#c84b31", label="before training")
    axes[0].plot(times, final_prediction, color="#2f6f9f", label="after training")
    axes[0].set(xlabel="time t", ylabel="y(t)", ylim=(0, 1.05))
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(range(1, 2001), theta_history, color="#2f6f9f")
    axes[1].axhline(0.8, ls="--", color="#202020", label="true rate")
    axes[1].set(xlabel="training update", ylabel="learned rate θ")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "02_decay_parameter_learning.pdf")
    plt.close(fig)
    return {"learned_theta": float(theta.detach()), "final_loss": loss_history[-1]}


def experiment_3() -> dict:
    times = torch.arange(0, 5.01, 0.5)
    observations = decay_solution(times).unsqueeze(-1)
    initial = torch.tensor([1.0])
    model = SmallMLP(1, 1)
    initial_prediction = euler_solve(model, initial, times, EULER_STEP).detach().squeeze()
    result = fit_trajectory(model, model.parameters(), initial, times, observations,
                            steps=2000, learning_rate=0.001, euler_step=EULER_STEP)
    save_history("03_decay_neural_loss", result.history)
    state_range = torch.linspace(float(observations.min()), 1.0, 150).unsqueeze(-1)
    with torch.no_grad():
        learned_derivative = model(state_range).squeeze()
    true_derivative = decay_derivative(state_range.squeeze())
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    axes[0].plot(times, observations.squeeze(), "ko", ms=3, label="observations")
    axes[0].plot(times, initial_prediction, "--", color="#c84b31", label="before training")
    axes[0].plot(times, result.predictions.squeeze(), color="#2f6f9f", label="after training")
    axes[0].set(xlabel="time t", ylabel="y(t)", ylim=(0, 1.05))
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(state_range.squeeze(), true_derivative, color="#202020", label="true: -0.8y")
    axes[1].plot(state_range.squeeze(), learned_derivative, color="#2f6f9f", label="learned fθ(y)")
    axes[1].set(xlabel="current value y", ylabel="derivative")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "03_decay_neural_rule.pdf")
    plt.close(fig)
    derivative_mse = torch.mean((learned_derivative - true_derivative) ** 2)
    return {"final_loss": result.history[-1], "derivative_mse_on_observed_range": float(derivative_mse)}


def experiment_4() -> dict:
    torch.manual_seed(SEED + 3)
    times = torch.arange(0, 16, 1.0)
    train_infected = [0.01, 0.10]
    observations = torch.stack([si_solution(times, infected0=value) for value in train_infected], dim=1)
    initial = torch.tensor([[0.99, 0.01], [0.90, 0.10]])
    model = SmallMLP(2, 2)
    result = fit_trajectory(model, model.parameters(), initial, times, observations,
                            steps=500, learning_rate=0.001, euler_step=SI_EULER_STEP)
    save_history("04_si_neural_loss", result.history)

    test_initial = torch.tensor([0.95, 0.05])
    true_test = si_solution(times, infected0=0.05)
    with torch.no_grad():
        predicted_test = euler_solve(model, test_initial, times, SI_EULER_STEP)
        physical_line = torch.linspace(0.01, 0.99, 150)
        states = torch.stack((1 - physical_line, physical_line), dim=-1)
        learned = model(states)
        true = si_derivative(states)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    axes[0].plot(times, true_test[:, 0], color="#2f6f9f", label="true S")
    axes[0].plot(times, true_test[:, 1], color="#c84b31", label="true I")
    axes[0].plot(times, predicted_test[:, 0], "--", color="#2f6f9f", label="predicted S")
    axes[0].plot(times, predicted_test[:, 1], "--", color="#c84b31", label="predicted I")
    axes[0].set(xlabel="time t", ylabel="population fraction", ylim=(-0.03, 1.03))
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].plot(physical_line, true[:, 1], color="#202020", label="true dI/dt")
    axes[1].plot(physical_line, learned[:, 1], color="#c84b31", label="learned dI/dt")
    axes[1].plot(physical_line, true[:, 0], color="#666666", ls="--", label="true dS/dt")
    axes[1].plot(physical_line, learned[:, 0], color="#2f6f9f", ls="--", label="learned dS/dt")
    axes[1].set(xlabel="infected fraction I (with S = 1 - I)", ylabel="derivative")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "04_si_neural_rule_and_new_initial_condition.pdf")
    plt.close(fig)

    rmse = torch.sqrt(torch.mean((predicted_test - true_test) ** 2))
    derivative_mse = torch.mean((learned - true) ** 2)
    return {
        "final_loss": result.history[-1],
        "test_trajectory_rmse_at_I0_0_05": float(rmse),
        "derivative_mse_on_S_plus_I_equals_1": float(derivative_mse),
    }


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    HISTORIES.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    setup()
    metrics = {
        "settings": {
            "seed": SEED, "euler_step_for_decay_experiments": EULER_STEP,
            "euler_step_for_SI_experiment": SI_EULER_STEP, "optimizer": "Adam",
            "learning_rate_for_decay_experiments": 0.001,
            "learning_rate_for_SI_experiment": 0.001,
            "updates_for_decay_experiments": 2000,
            "updates_for_SI_experiment": 500,
            "SI_note": "The SI example uses h = 0.2 and 500 updates instead of h = 0.01 and 2000 updates because differentiating through 1500 Euler steps per update was too slow on the available CPU. The purpose is an understandable teaching demonstration, not hyperparameter optimization.",
        },
        "experiment_1_euler": experiment_1(),
        "experiment_2_parameter_learning": experiment_2(),
        "experiment_3_neural_decay": experiment_3(),
        "experiment_4_neural_si": experiment_4(),
    }
    save_metrics(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
