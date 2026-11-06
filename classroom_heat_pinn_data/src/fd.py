"""Wrapper around the existing finite-volume solver (synthetic/solver.py).

Arrays keep the solver convention theta[ix, iy] (indexing='ij'); never transpose
for computation. Heat-flow sign convention: F_i > 0 is outflow through wall i.
"""
import numpy as np

try:
    from ..synthetic import solver
    from . import common
except ImportError:  # run as `python -m src.<module>` from classroom_heat_pinn_data/
    from synthetic import solver
    import src.common as common

INVERSE_NX = 101   # 101 x 81 grid for fitting (reference uses 201 x 161)
FINE_NX = 201


def solve_field(Bi, known, nx=INVERSE_NX, heater_key="heater"):
    """Return x, y, theta[ix, iy], info for the simple model with the given Bi."""
    return solver.solve(nx, common.bi_dict(Bi), common.theta_inf(known),
                        heaters=common.heaters(known, heater_key))


def sample(x, y, theta, xy):
    """Bilinear interpolation at points xy (n, 2) of a field theta[ix, iy]."""
    return solver.sample(x, y, theta, np.asarray(xy, float))


def node_indices(x, y, xy, tol=1e-9):
    """Grid node (ix, iy) for each point, or None if a point is not on a node."""
    ix = np.rint(xy[:, 0] / (x[1] - x[0])).astype(int)
    iy = np.rint(xy[:, 1] / (y[1] - y[0])).astype(int)
    if (np.abs(x[ix] - xy[:, 0]).max() > tol) or (np.abs(y[iy] - xy[:, 1]).max() > tol):
        return None
    return ix, iy


def heat_flows(info):
    """Signed wall flows from solver info (discrete FV integrals).

    F_i = int Bi_i (theta - theta_inf_i) ds (outflow positive), H = int Q dA.
    share_signed_i = F_i / H may be negative (inflow) or exceed 1; never clipped.
    share_outflow_i = max(F_i, 0) / sum_j max(F_j, 0).
    """
    F = {w: float(info["total_flux"][w]) for w in common.ROBIN_WALLS}
    H = float(info["q_in"])
    out = {w: max(F[w], 0.0) for w in F}
    out_sum = sum(out.values())
    total = sum(F.values()) + float(info["total_flux"]["W"])
    return {
        "flux": F,
        "source_integral": H,
        "share_signed": {w: F[w] / H for w in F},
        "share_outflow": {w: (out[w] / out_sum if out_sum > 0 else None) for w in F},
        "inflow": {w: min(F[w], 0.0) for w in F},
        "energy_balance_relerr": abs(total - H) / max(abs(H), 1e-12),
        "ranking": sorted(F, key=lambda w: -F[w]),
    }
