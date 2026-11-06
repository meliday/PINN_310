"""Truth-based evaluation. The only src module allowed to open truth files.

Every truth read is appended to results/logs/truth_access.log with its purpose.
"""
import datetime as _dt
import json

import numpy as np
from scipy.spatial import Delaunay

try:
    from . import common, fd
except ImportError:
    import src.common as common
    import src.fd as fd

WALL_BAND = 0.05
METRIC_COLUMNS = [
    "exp_id", "method", "data_file", "seed", "status", "n_unknown",
    "bi_S_hat", "bi_E_hat", "bi_N_hat", "A_hat",
    "bi_S_relerr", "bi_E_relerr", "bi_N_relerr",
    "flux_S", "flux_E", "flux_N", "source_integral",
    "share_S_err_pp", "share_E_err_pp", "share_N_err_pp", "ranking_correct",
    "q1_rmse_C", "q1_maxabs_C", "q1_wall_rmse_C", "q3_rmse_C", "q3_coldest_err_C",
    "loo_rmse_C", "loo_mae_C", "energy_balance_relerr", "wall_time_s", "notes",
]


def _log_access(fname, purpose, log_dir=None):
    log_dir = log_dir or common.RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    with open(log_dir / "truth_access.log", "a") as fh:
        fh.write(f"{stamp}\t{fname}\t{purpose}\n")


def load_truth_config(purpose, data_dir=common.DATA_DIR):
    _log_access("config.json", purpose)
    return json.loads((data_dir / "config.json").read_text())


def load_truth_bi(purpose, data_dir=common.DATA_DIR):
    return common.bi_dict(load_truth_config(purpose, data_dir)["Bi_true"])


def load_reference(purpose, kind="reference", data_dir=common.DATA_DIR):
    fname = f"{kind}.npz"
    if fname not in common.TRUTH_FILES:
        raise ValueError(fname)
    _log_access(fname, purpose)
    with np.load(data_dir / fname) as ref:
        return {k: ref[k] for k in ref.files}


def q1_metrics(theta_hat, ref, obs, known):
    """Q1 errors in C on the reference grid, excluding training-sensor nodes.

    theta_hat: prediction on the reference grid, shape (nx, ny), [ix, iy].
    """
    x, y, theta = ref["x"], ref["y"], ref["theta"]
    if theta_hat.shape != theta.shape:
        raise ValueError(f"shape {theta_hat.shape} != reference {theta.shape}")
    X, Y = np.meshgrid(x, y, indexing="ij")
    mask = np.ones(theta.shape, bool)
    nodes = fd.node_indices(x, y, obs.xy)
    if nodes is None:
        raise ValueError("sensors are not on reference grid nodes")
    mask[nodes] = False
    if "fluid" in ref:  # X1/X2 references: nodes inside solids are NaN and not evaluated
        mask &= ref["fluid"].astype(bool)
    LX, LY = known["domain"]["LX"], known["domain"]["LY"]
    band = np.minimum.reduce([X, LX - X, Y, LY - Y]) <= WALL_BAND
    inside = Delaunay(obs.xy).find_simplex(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape) >= 0
    err = (theta_hat - theta) * common.dt_K(known)

    def rmse(m):
        return float(np.sqrt(np.mean(err[m] ** 2))) if m.any() else None

    out = {
        "q1_rmse_C": rmse(mask),
        "q1_maxabs_C": float(np.abs(err[mask]).max()),
        "q1_wall_rmse_C": rmse(mask & band),
        "q1_rmse_in_hull_C": rmse(mask & inside),
        "q1_rmse_out_hull_C": rmse(mask & ~inside),
        "frac_out_hull": float((mask & ~inside).sum() / mask.sum()),
        "n_eval_points": int(mask.sum()),
    }
    if "wall_tip" in ref:  # re-entrant wall tip: reported separately (numerical uncertainty is largest there)
        tip = ref["wall_tip"].astype(bool)
        out["q1_rmse_wall_tip_C"] = rmse(mask & tip)
        out["q1_rmse_outside_wall_tip_C"] = rmse(mask & ~tip)
    return out


def true_heat_flows(known, purpose, nx=fd.FINE_NX):
    """Heat flows of the true simple model on the fine grid (evaluation only)."""
    bi_true = load_truth_bi(purpose)
    _, _, _, info = fd.solve_field(bi_true, known, nx=nx)
    return bi_true, fd.heat_flows(info)


def bi_and_flux_metrics(bi_hat, flows_hat, bi_true, flows_true):
    out = {}
    for w in common.ROBIN_WALLS:
        out[f"bi_{w}_hat"] = bi_hat[w]
        out[f"bi_{w}_relerr"] = (bi_hat[w] - bi_true[w]) / bi_true[w]
        out[f"flux_{w}"] = flows_hat["flux"][w]
        out[f"share_{w}_err_pp"] = 100 * (flows_hat["share_signed"][w] - flows_true["share_signed"][w])
    out["source_integral"] = flows_hat["source_integral"]
    out["ranking_correct"] = flows_hat["ranking"] == flows_true["ranking"]
    out["energy_balance_relerr"] = flows_hat["energy_balance_relerr"]
    return out


def metrics_row(**values):
    row = {c: None for c in METRIC_COLUMNS}
    unknown = set(values) - set(METRIC_COLUMNS)
    if unknown:
        raise KeyError(f"not metric columns: {sorted(unknown)}")
    row.update(common.to_jsonable(values))
    return row
