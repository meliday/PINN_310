# Classroom heat loss from 10 sensors: PINN vs FD-inverse vs RBF

Estimate the full temperature field of a room and the heat-loss coefficient of each wall from a handful of temperature sensors, and compare three approaches on the same data:

- **PINN** — a physics-informed neural network (4 × 64 tanh, float64) that learns the field and the wall coefficients together.
- **FD-inverse** — a finite-volume forward solver wrapped in nonlinear least squares over the coefficients.
- **RBF** — thin-plate-spline interpolation of the sensors (temperature only, no physics).

Everything below is on **synthetic data** from a simplified 2D steady heat-conduction model, with one noise realisation and one PINN seed. It is a method comparison, not a measurement of a real classroom.

**Bottom line:** for this problem FD-inverse is the baseline to beat. It recovered the coefficients in every condition tested, in 2–4 s per fit. The PINN matched FD on the basic room, was not clearly better anywhere, and failed on two of the extensions (a local cooling sink and an internal wall). The full write-up is [`classroom_heat_pinn_data/results/REPORT.md`](classroom_heat_pinn_data/results/REPORT.md).

## The model

A 10 m × 8 m room, nondimensionalised to x ∈ [0, 1], y ∈ [0, 0.8], θ = (T − 0 °C) / 20 K:

- −Δθ = Q in the room, with a Gaussian heater at (2 m, 6 m).
- Robin walls −∂θ/∂n = Bi (θ − θ∞): S window (θ∞ = 0), E exterior (θ∞ = 0), N corridor (θ∞ = 0.7, i.e. 14 °C).
- Adiabatic W wall (shared with the next classroom).
- Unknowns: the three Biot numbers Bi_S, Bi_E, Bi_N (truth 1.0, 0.4, 0.25).
- Ten sensors (layout A), clean or with σ = 0.1 °C Gaussian noise.

Extensions requested during the project:

- **E5b** — a cooling sink at a known position (7.5 m, 0.3 m) removing an unknown fraction f of the heater power (truth f = 0.1).
- **X1** — an adiabatic internal wall (x ∈ [0, 6] m, y ∈ [3.0, 3.4] m) leaving a 4 m passage on the east side.
- **X2** — four conditions on the same sensors: base / sink only / wall only / wall + sink.

## How it works

### Pipeline

A generator solves the true model on a fine grid and splits its output into what the estimators may see (room geometry and sensor readings) and what only the evaluator may see (true coefficients and the full field). PINN, FD-inverse and RBF all fit from the same sensor file; the truth is read only after fitting, and every read is logged.

### PINN

One network predicts temperature everywhere. The physics is enforced softly through loss terms, and the unknown coefficients are trained alongside the network weights.

- The network maps a point (x, y) to θ; derivatives come from autograd.
- The loss sums three residuals: the PDE inside the room (−Δθ − Q), the wall conditions (∂θ/∂n + Bi (θ − θ∞)), and the misfit at the 10 sensors.
- Adam (20000 steps) and then L-BFGS (100 iterations) update the weights together with log Bi (and log f when there is a sink).

Collocation points are resampled every 500 steps: 1000 inside the room (30% near the heater) and 100 per wall. With an internal wall, points inside the solid are dropped and its faces get a no-flux residual. This loop is where the PINN failed:

- **Sink (E5b):** shrinking f is the easiest way to lower the PDE loss early in training, so f collapses to zero.
- **Wall (X1):** the network smooths across the wall instead of resolving the temperature jump.

### FD-inverse

The physics is solved exactly on a grid; only the few coefficients are searched.

- Start from 3–4 initial guesses for log Bi (and log f).
- For each guess: solve the finite-volume problem on a 101 × 81 grid, interpolate at the sensors, and compare with the readings.
- Trust-region least squares, within bounds, updates the guess until it converges. The converged start with the lowest cost wins, and its Jacobian gives the local uncertainty.

Each solve takes milliseconds, so a full fit with several starting points finishes in seconds. Internal walls are handled by removing solid grid cells, which makes their faces exactly no-flux.

## Results

Temperature error is the RMSE over grid nodes that are not training sensors (°C). Bi errors are relative errors in S / E / N order.

| Condition | Data | Method | Bi error S / E / N | f error | Temperature RMSE °C | Fit time |
|---|---|---|---|---|---|---|
| Base | clean | FD-inverse | ≈ 0 / ≈ 0 / −0.1% | — | 0.0005 | 2 s |
| Base | clean | PINN (Adam 20000 + L-BFGS) | +0.03 / +0.23 / −0.16% | — | 0.0068 | 340 s |
| Base | clean | RBF | — | — | 0.513 | < 0.01 s |
| Base | noisy | FD-inverse | −2.0 / +3.5 / +5.2% | — | 0.045 | 3 s |
| Base | noisy | PINN | −1.8 / +3.2 / +5.3% | — | 0.042 | 432 s |
| Model mismatch (E5) | noisy | FD / PINN | both ≈ +31% on Bi_E | — | 0.364 / 0.354 | — |
| Sink (E5b) | clean | FD-inverse | ≈ 0 | −0.04% | 0.0005 | 3 s |
| Sink (E5b) | clean | **PINN** | +15 / +19 / −25% | **−99.5% (f → 0)** | 0.181 | 334 s |
| Wall (X1) | clean | FD-inverse, grid 201 | +0.24 / −0.12 / −0.43% | — | 0.004 | < 30 s |
| Wall (X1) | none (true Bi fixed) | **PINN forward** | — | — | **13.3 (rel. L2 54%), all fluid nodes** | 342 s |
| Wall + sink (X2) | clean | FD-inverse, grid 101 | +0.8 / −0.3 / −1.3% | −0.4% | 0.022 | — |
| Wall + sink (X2) | noisy | FD-inverse, grid 101 | +2.2 / −1.3 / −1.6% | −3.2% | 0.048 | — |

What the results show:

- **The PINN failures are optimisation failures, not representation limits.**
  - With the true sink strength fixed, the same PINN learns the E5b field to 0.009 °C. Learning the field and the coefficient together drives f to zero.
  - With the true Bi fixed, the X1 PINN still smooths straight across the 0.4 m wall: a jump of more than 25 °C that it never resolves.
- **Leaving physics out of the model biases the coefficients, sometimes invisibly.**
  - Fitting the wall data without the wall collapses Bi_E and Bi_N to their lower bounds, and the temperature error reaches 6.6 °C.
  - Fitting sink data without the sink still matches temperatures to about 0.2 °C, but Bi is off by up to 27%. A good temperature fit does not mean correct coefficients.
- **All PINN changes were pre-registered with a fixed attempt limit and stopped there.** After the E5b and X1 failures, no further PINN work was done.

## Repository layout

All research code lives in `classroom_heat_pinn_data/`.

**Data and solvers** — `synthetic/`

| Path | Contents |
|---|---|
| `solver.py` | Vertex-centred finite-volume solver for the rectangular room |
| `solver_wall.py` | The same solver with internal adiabatic walls (solid cells removed) |
| `generate_data.py`, `generate_e5b.py`, `generate_x1.py`, `generate_x2.py` | Dataset generators: base, sink, wall, wall + sink |
| `data/`, `e5b/`, `x1/`, `x2/` | Generated datasets: `known_*.json` and `sensors_*.csv` for training; `config_*.json` and `reference_*.npz` are ground truth |

**Estimation and experiments** — `src/`

| Path | Contents |
|---|---|
| `common.py` | Training-side loaders; never open truth files |
| `evaluate.py` | The only truth loader; logs every read |
| `fd.py`, `baselines.py` | FD forward wrapper, FD-inverse, RBF interpolation |
| `pinn.py` | PINN model, collocation sampling, losses, training loop |
| `run_experiments.py` | Command-line entry point for every experiment |
| `extensions_fd.py` | FD runs for X1 and X2 |

**Everything else**

| Path | Contents |
|---|---|
| `classroom_heat_pinn_data/results/` | Per-experiment summaries, checkpoints, figures, `metrics.csv`, `REPORT.md` |
| `classroom_heat_pinn_data/diagnostics/` | Development-only PINN and solver diagnostics |
| `classroom_heat_pinn_data/real/` | Cleaned public indoor-temperature data (Medellín ITM, Intel Berkeley Lab); not used in the results above |
| `tests/` | `unittest` suite (125 tests) |
| `hackathon-demo/` | Interactive web demo (not tracked by git) |

## Setup

Tested with Python 3.12.13, PyTorch 2.13.0, NumPy 2.5.1, SciPy 1.18.0, pandas 3.0.3, matplotlib 3.11.0, on CPU in float64. The project uses the standard-library `unittest`; pytest is not required.

```bash
python -B -m unittest discover -s tests
```

## Running experiments

From the repository root:

```bash
python -B -m classroom_heat_pinn_data.src.run_experiments --exp E0
python -B -m classroom_heat_pinn_data.src.run_experiments --exp E2 --data A_clean --seed 0 --config C1
python -B -m classroom_heat_pinn_data.src.run_experiments --exp E5b_fd
python -B -m classroom_heat_pinn_data.src.run_experiments --exp X2_fd
```

Experiment IDs: `E0` (FD + RBF baseline), `E1` (PINN forward, true Bi), `E2` (PINN inverse), `E3` (fixed config on noisy data), `E5` (model mismatch), `E5b_ident`, `E5b_fd`, `E5b_pinn`, `X1_forward`, `X1_fd`, `X2_fd`.

- Each run writes to its own folder under `results/` and upserts a row into `results/metrics.csv`.
- An existing run is refused with exit code 3 unless `--overwrite` is passed. All committed results already exist, so rerunning needs `--overwrite` or a fresh checkout.
- PINN runs take about 6 minutes each on CPU; FD runs take seconds.

To regenerate a dataset, write to a new folder; existing folders are refused:

```bash
python -B -m classroom_heat_pinn_data.synthetic.generate_x2 --output-dir /tmp/x2_new
```

## Truth isolation

The fitting code reads only `known*.json` and `sensors_*.csv`. Ground-truth files are read only by evaluation code, after fitting, and each read is appended to `results/logs/truth_access.log`. Tests check this with Python audit hooks, including a symlink positive control.

The X1/X2 truth fields are solved on a 1601-grid and stored at the 201-grid nodes. The re-entrant wall tip limits grid convergence, so wall-tip errors are reported separately.

## Interactive demo

`hackathon-demo/` is a local web page plus a small loopback server. It shows the saved PINN and FD predictions for all four rooms, the controls that leave physics out, and a live FD re-solve when you drag the heater (about 0.1 s per request).

```bash
python -B hackathon-demo/server.py --port 8765   # then open http://127.0.0.1:8765
```

The folder is not committed. Its extension data are exported from the research working tree, with SHA-256 hashes recorded in `hackathon-demo/data/extensions-verification.json`. Restart the server after changing `server.py`, because a running process keeps the old code.

## Limitations

- Synthetic data only, with one noise realisation per condition, one sensor layout and one PINN seed. No claim of stability or superiority follows from these runs.
- X1/X2 keep the heater amplitude fixed, so the region above the wall reaches about 40 °C, which is unrealistic for a classroom.
- Not done:
  - comparing the four conditions at equal effective heat input;
  - estimating the sink position;
  - repeated noise draws;
  - experiments on the real data in `real/`;
  - PINN inverse runs for X1/X2.

## Project history

Development was split between agents, with every step logged and cross-reviewed:

- **Claude** implemented most of the code and ran most of the experiments.
- **Codex** independently verified and reviewed the work, ran the PINN failure diagnostics, and contributed fixes.

Experiments were pre-registered before running, and failures are reported as they occurred. The detailed log (`AGENT_codexandclaude.md`) and plan (`PROJECT_PLAN.md`) are kept locally and are not part of the repository.
