# Neural ODE bachelor seminar

This repository contains the report, the original seminar material, and the
small experiments used in the report.

The latest report is [main.tex](main.tex) in the repository root. The earlier
draft is kept unchanged at
[report/archive/main-before-revision.tex](report/archive/main-before-revision.tex).
The original papers and presentation are preserved in references/papers and
presentation. The copies in the repository root were left in place.

## Layout

- main.tex is the current LaTeX report source.
- report contains the generated figures and the archived earlier draft.
- notebooks contains the teaching walkthroughs in the intended reading order.
- src/neural_odes_seminar contains the small reusable implementation.
- results contains generated losses, parameter histories, and metrics.
- tests contains basic checks of solutions, Euler accuracy, and gradients.

## Reproducing the experiments

Use Python 3.13 or later. Install PyTorch's CPU build, then the remaining
packages:

    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    python -m pip install numpy matplotlib jupyter pytest

From the repository root, generate all figures and result files:

    $env:PYTHONPATH = "src"
    python -m neural_odes_seminar.run

The runner uses fixed seed 7. It writes PDF figures to report/figures, loss
histories to results/histories, and summary metrics to
results/metrics/experiment_metrics.json.

The SI example uses an Euler step size of 0.2 and 500 training updates. This
is a documented runtime adjustment: differentiating through the originally
planned 0.01-step solve was too slow for a small CPU teaching project. The
decay experiments retain step size 0.01 and 2,000 updates.

To run the checks:

    $env:PYTHONPATH = "src"
    python -m pytest

## Notebooks and report

Open the notebooks in this order:

1. 01_understanding_decay.ipynb
2. 02_learning_decay.ipynb
3. 03_learning_si.ipynb

They call the shared implementation rather than copying it. Compile main.tex
from the repository root with a standard LaTeX installation after generating
the figures.
