# GradCell

GradCell learns a goal-conditioned initializer and a small number of physics-gradient
updates for battery-cell design. The MVP uses a seven-dimensional hard-feasible design
space, a PyTorch custom autograd layer, PyBaMM SPMe/IDAKLU forward sensitivities, and
DFN as a later verification backend.

## What is implemented

- hard-feasible decoder for porosity, active fraction, N/P ratio, and diffusivity multipliers;
- capacity, C-rate current, and stack-mass calculations;
- interchangeable analytic-test and PyBaMM physics backends;
- custom `torch.autograd.Function` performing `J^T v` in backward;
- smooth voltage-cutoff energy and power metrics;
- goal-conditioned initializer and learned diagonal physics refiner;
- Smooth Tchebycheff preference loss;
- directional gradient validation, MVP training, and unit tests.

The analytic backend is a software test fixture, not a scientific battery model. All
scientific conclusions must use the PyBaMM backend and later DFN verification.

## Repository layout

```text
configs/                 experiment configuration
src/gradcell/design/     feasible decoder, capacity, mass
src/gradcell/physics/    PyBaMM backend, custom autograd, metrics, gradient QA
src/gradcell/models/     preference encoder, initializer, refiner, GradCell
src/gradcell/losses/     multi-objective scalarization
src/gradcell/training/   task sampling and curriculum-ready trainer
src/gradcell/benchmark/  regret and future reference-front code
scripts/                 runnable experiment entry points
tests/                   fast unit and gradient tests
```

## Installation

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[physics,dev]"
```

Use Python 3.10 or 3.11 and keep the first experiments on CPU with `torch.float64`.

## First experiment sequence

Run the fast tests first:

```powershell
pytest
python scripts/validate_gradients.py --backend toy
python scripts/train_mvp.py --backend toy --steps 100 --refinement-steps 0
```

Then validate the real physics backend:

```powershell
python scripts/run_baseline.py --c-rate 1
python scripts/validate_gradients.py --backend pybamm --eps 1e-4
```

Only after the PyBaMM directional gradient error is acceptable should online training
start:

```powershell
python scripts/train_mvp.py --backend pybamm --steps 5000 --batch-size 2 --refinement-steps 0
python scripts/train_mvp.py --backend pybamm --steps 5000 --batch-size 2 --refinement-steps 1
python scripts/train_mvp.py --backend pybamm --steps 20000 --batch-size 2 --refinement-steps 3
```

Recommended Go/No-Go thresholds before training:

- all nominal SPMe simulations finish without NaN;
- median directional gradient error below `1e-3` in smooth interior points;
- 95th percentile directional error below `1e-2`;
- a small exact-gradient step lowers the objective for most interior designs;
- no solver failures are silently removed.

## Current research boundary

The main variables are non-geometric because standard PyBaMM meshes do not make particle
radius or electrode thickness safe runtime inputs. The MVP therefore uses positive and
negative solid-diffusivity multipliers instead of particle radii. Radius and thickness
should be added only after a normalized-coordinate backend or a separately validated
finite-difference path exists.
Training uses fixed horizons and a smooth voltage gate; hard voltage cutoff and DFN are
evaluation checks rather than the training derivative path.
