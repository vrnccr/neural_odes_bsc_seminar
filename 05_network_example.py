"""Create the small network example used in the report.

The script intentionally does not depend on the Neural ODE experiment code.
It illustrates the adjacency matrix and the product A i for one fixed
four-node contact network.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "report" / "figures" / "05_network_example.pdf"


def main() -> None:
    adjacency = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0]],
        dtype=int,
    )
    infected = np.array([0, 1, 0, 0], dtype=int)
    infected_neighbours = adjacency @ infected
    assert np.array_equal(infected_neighbours, np.array([1, 0, 1, 1]))

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, (graph_axis, matrix_axis) = plt.subplots(1, 2, figsize=(9.0, 4.1))

    positions = {
        1: (0.15, 0.5),
        2: (0.5, 0.5),
        3: (0.85, 0.78),
        4: (0.85, 0.22),
    }
    for first, second in [(1, 2), (2, 3), (2, 4)]:
        x_values = [positions[first][0], positions[second][0]]
        y_values = [positions[first][1], positions[second][1]]
        graph_axis.plot(x_values, y_values, color="#555555", linewidth=1.8, zorder=1)

    for node, (x_value, y_value) in positions.items():
        infected_node = node == 2
        colour = "#c84b31" if infected_node else "#d9e8f5"
        state = "I" if infected_node else "S"
        graph_axis.scatter(
            x_value,
            y_value,
            s=1_350,
            color=colour,
            edgecolor="#333333",
            linewidth=1.2,
            zorder=2,
        )
        graph_axis.text(
            x_value,
            y_value,
            f"{node}\n{state}",
            ha="center",
            va="center",
            fontweight="bold",
            zorder=3,
        )

    graph_axis.set_title("Contact graph")
    graph_axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    graph_axis.set_aspect("equal")
    graph_axis.axis("off")
    graph_axis.text(
        0.5,
        0.03,
        "Node 2 is infected; nodes 1, 3, and 4 are susceptible.",
        ha="center",
        va="bottom",
        transform=graph_axis.transAxes,
        fontsize=9,
    )

    matrix_axis.axis("off")
    matrix_axis.set_title("Adjacency matrix and node states")
    matrix_axis.text(
        0.05,
        0.88,
        r"$A=$",
        transform=matrix_axis.transAxes,
        fontsize=15,
        va="center",
    )
    matrix_table = matrix_axis.table(
        cellText=adjacency,
        rowLabels=["1", "2", "3", "4"],
        colLabels=["1", "2", "3", "4"],
        cellLoc="center",
        rowLoc="center",
        bbox=[0.19, 0.52, 0.36, 0.40],
    )
    matrix_table.auto_set_font_size(False)
    matrix_table.set_fontsize(10)

    matrix_axis.text(
        0.64,
        0.76,
        r"$\mathbf{i}=$",
        transform=matrix_axis.transAxes,
        fontsize=15,
        va="center",
    )
    vector_i = matrix_axis.table(
        cellText=[[value] for value in infected],
        cellLoc="center",
        bbox=[0.83, 0.54, 0.10, 0.35],
    )
    vector_i.auto_set_font_size(False)
    vector_i.set_fontsize(10)

    matrix_axis.text(
        0.05,
        0.27,
        r"$A\mathbf{i}=$",
        transform=matrix_axis.transAxes,
        fontsize=15,
        va="center",
    )
    vector_ai = matrix_axis.table(
        cellText=[[value] for value in infected_neighbours],
        cellLoc="center",
        bbox=[0.29, 0.08, 0.10, 0.35],
    )
    vector_ai.auto_set_font_size(False)
    vector_ai.set_fontsize(10)
    matrix_axis.text(
        0.48,
        0.20,
        "Nodes 1, 3, and 4\nhave an infected neighbour.",
        transform=matrix_axis.transAxes,
        fontsize=10,
        va="center",
    )

    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, format="pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
