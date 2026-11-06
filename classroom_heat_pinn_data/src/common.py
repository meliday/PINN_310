"""Training-side data access: known config and sensor observations only.

Nothing in this module may open ground-truth files (config.json, *reference.npz).
Truth loading lives in `evaluate.py` and is only called from evaluation paths.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PKG_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PKG_DIR / "synthetic" / "data"
RESULTS_DIR = PKG_DIR / "results"

# Files an estimation method may read. Everything else in DATA_DIR is truth.
TRAINING_FILES = {
    "known.json",
    "sensors_A_clean.csv", "sensors_A_noisy.csv",
    "sensors_B_clean.csv", "sensors_B_noisy.csv",
    "sensors_A_mismatch.csv",
}
TRUTH_FILES = {
    "config.json", "reference.npz", "whatif_reference.npz", "mismatch_reference.npz",
}

ROBIN_WALLS = ("S", "E", "N")
SENSOR_COLUMNS = ["sensor_id", "x", "y", "theta_obs", "T_obs_C"]
CSV_T_ATOL_C = 1e-4


@dataclass(frozen=True)
class Observations:
    name: str
    sensor_id: np.ndarray
    xy: np.ndarray        # (n, 2), nondimensional (x, y)
    theta: np.ndarray     # (n,)
    T_C: np.ndarray       # (n,)


def load_known(data_dir=DATA_DIR):
    known = json.loads((Path(data_dir) / "known.json").read_text())
    if any(key.lower().startswith("bi") for key in known):
        raise ValueError("known.json unexpectedly contains Bi values")
    return known


def dt_K(known):
    return float(known["temperature_scaling"]["DT_K"])


def load_observations(name, known, data_dir=DATA_DIR):
    """Load sensors_<name>.csv, e.g. name='A_clean'. Checks columns and units."""
    fname = f"sensors_{name}.csv"
    if fname not in TRAINING_FILES:
        raise ValueError(f"{fname} is not an allowed training file")
    df = pd.read_csv(Path(data_dir) / fname)
    if list(df.columns) != SENSOR_COLUMNS:
        raise ValueError(f"{fname}: columns {list(df.columns)} != {SENSOR_COLUMNS}")
    xy = df[["x", "y"]].to_numpy(float)
    theta = df["theta_obs"].to_numpy(float)
    T_C = df["T_obs_C"].to_numpy(float)
    LX, LY = known["domain"]["LX"], known["domain"]["LY"]
    if not (np.all((xy[:, 0] >= 0) & (xy[:, 0] <= LX)) and np.all((xy[:, 1] >= 0) & (xy[:, 1] <= LY))):
        raise ValueError(f"{fname}: sensor outside the domain")
    # CSV rounds theta and T separately; compare with an absolute tolerance only.
    T_out = known["temperature_scaling"]["T_out_C"]
    err = np.abs(T_C - (T_out + dt_K(known) * theta)).max()
    if err > CSV_T_ATOL_C:
        raise ValueError(f"{fname}: T_obs_C differs from DT*theta by {err:.2e} C")
    return Observations(name, df["sensor_id"].to_numpy(), xy, theta, T_C)


def heaters(known, key="heater"):
    """Solver heater list from known.json ('heater' or 'heater_whatif')."""
    h = known[key]
    return [{"x": h["x"], "y": h["y"], "A": h["A"], "s": h["s"]}]


def theta_inf(known):
    return {w: float(known["theta_inf"][w]) for w in ROBIN_WALLS}


def bi_dict(bi):
    """Accept a dict keyed by S/E/N or a length-3 sequence in S, E, N order."""
    if isinstance(bi, dict):
        return {w: float(bi[w]) for w in ROBIN_WALLS}
    bi = list(bi)
    if len(bi) != 3:
        raise ValueError("Bi must have three values (S, E, N)")
    return dict(zip(ROBIN_WALLS, map(float, bi)))


def to_jsonable(obj):
    """Convert numpy scalars/arrays; refuse NaN/Inf so they never reach results."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
