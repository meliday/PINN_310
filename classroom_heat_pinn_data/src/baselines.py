"""Baselines: RBF interpolation (Q1 only) and FD-inverse (Q1-Q3).

Both take observations and known.json only; neither touches truth files.
"""
import time

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import least_squares

try:
    from . import common, fd
except ImportError:
    import src.common as common
    import src.fd as fd

BI_BOUNDS = (0.01, 50.0)
DEFAULT_STARTS = ((1.0, 1.0, 1.0), (0.1, 0.1, 0.1), (5.0, 5.0, 5.0))


def reconstruct_rbf(obs_xy, obs_theta, query_xy):
    """Thin-plate spline interpolation, smoothing=0. Q1 only (no Bi, flux, what-if)."""
    rbf = RBFInterpolator(np.asarray(obs_xy, float), np.asarray(obs_theta, float),
                          kernel="thin_plate_spline", smoothing=0.0)
    return rbf(np.asarray(query_xy, float))


def fit_fd_inverse(obs, known, experiment=None):
    """Least-squares fit of beta = log Bi (S, E, N) to sensor theta on the 101 x 81 grid.

    experiment: optional dict with 'starts' (Bi triples), 'nx', 'max_nfev'.
    Runs every start, keeps all results, and selects the lowest training cost.
    """
    experiment = experiment or {}
    nx = experiment.get("nx", fd.INVERSE_NX)
    starts = experiment.get("starts", DEFAULT_STARTS)
    max_nfev = experiment.get("max_nfev", 200)
    lo, hi = np.log(BI_BOUNDS[0]), np.log(BI_BOUNDS[1])
    n_solves = [0]

    def predict(beta):
        n_solves[0] += 1
        x, y, theta, _ = fd.solve_field(np.exp(beta), known, nx=nx)
        return fd.sample(x, y, theta, obs.xy)

    def residual(beta):
        return predict(beta) - obs.theta

    runs = []
    t_all = time.perf_counter()
    for start in starts:
        t0, s0 = time.perf_counter(), n_solves[0]
        beta0 = np.clip(np.log(np.asarray(start, float)), lo, hi)
        try:
            res = least_squares(residual, beta0, bounds=(lo, hi), method="trf",
                                x_scale=1.0, ftol=1e-14, xtol=1e-14, gtol=1e-14,
                                max_nfev=max_nfev)
            runs.append({
                "start": list(start), "ok": bool(np.all(np.isfinite(res.x))),
                "Bi": dict(zip(common.ROBIN_WALLS, np.exp(res.x))),
                "beta": res.x, "cost": float(res.cost), "status": int(res.status),
                "message": res.message, "nfev": int(res.nfev),
                "n_solves": n_solves[0] - s0, "jac": res.jac,
                "time_s": time.perf_counter() - t0,
            })
        except Exception as exc:  # keep failures in the record
            runs.append({"start": list(start), "ok": False, "error": repr(exc),
                         "n_solves": n_solves[0] - s0, "time_s": time.perf_counter() - t0})
    fit_time = time.perf_counter() - t_all

    ok = [r for r in runs if r["ok"]]
    if not ok:
        return {"status": "failed", "runs": runs, "wall_time_s": fit_time}
    best = min(ok, key=lambda r: r["cost"])
    resid = residual(best["beta"])
    J = best["jac"]  # d theta_sensor / d beta at the selected solution
    sv = np.linalg.svd(J, compute_uv=False)
    rank = int(np.sum(sv > sv[0] * 1e-10))
    return {
        "status": "completed",
        "Bi": best["Bi"],
        "beta": best["beta"],
        "selected_start": best["start"],
        "sensor_rms_C": float(np.sqrt(np.mean(resid ** 2)) * common.dt_K(known)),
        "sensor_maxabs_C": float(np.abs(resid).max() * common.dt_K(known)),
        "jacobian": {"rank": rank, "singular_values": sv,
                     "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else None},
        "runs": [{k: v for k, v in r.items() if k != "jac"} for r in runs],
        "total_solves": n_solves[0],
        "wall_time_s": fit_time,
        "nx": nx,
    }
