"""Wrapper around the existing finite-volume solver (synthetic/solver.py).

Arrays keep the solver convention theta[ix, iy] (indexing='ij'); never transpose
for computation. Heat-flow sign convention: F_i > 0 is outflow through wall i.
"""
import numpy as np

try:
    from ..synthetic import solver, solver_wall
    from . import common
except ImportError:  # run as `python -m src.<module>` from classroom_heat_pinn_data/
    from synthetic import solver, solver_wall
    import src.common as common

INVERSE_NX = 101   # 101 x 81 grid for fitting (reference uses 201 x 161)
FINE_NX = 201


def solve_field(Bi, known, nx=INVERSE_NX, heater_key="heater", sink_fraction=None):
    """Return x, y, theta[ix, iy], info for the simple model with the given Bi.

    sink_fraction (E5b): if given, subtract the known-position Gaussian sink from known["sink"]
    with an amplitude that removes exactly that fraction of the heater power on this grid's
    control volumes (same construction as synthetic/generate_e5b.py). info then also carries
    heater_power and sink_power. Without it the base model is solved unchanged.
    known["solids"] (X1/X2): internal adiabatic walls, solved with synthetic/solver_wall.py on the
    fluid control volumes (theta is NaN inside solids); the sink fraction is of the fluid heater power.
    """
    solids = common.solids(known)
    if sink_fraction is None and not solids:
        return solver.solve(nx, common.bi_dict(Bi), common.theta_inf(known),
                            heaters=common.heaters(known, heater_key))
    x, y, _ = solver.make_grid(nx)
    X, Y = np.meshgrid(x, y, indexing="ij")
    vol = solver_wall.fluid_volumes(x, y, solids) if solids else _fv_volumes(x, y)
    Q = solver.heater_source(X, Y, common.heaters(known, heater_key))
    heater_power = float((Q * vol).sum())
    if sink_fraction is not None:
        geom = common.sink_geometry(known)
        if geom is None:
            raise ValueError("sink_fraction given but known has no sink geometry")
        shape = np.exp(-((X - geom["x"]) ** 2 + (Y - geom["y"]) ** 2) / (2 * geom["s"] ** 2))
        Q = Q - float(sink_fraction) * heater_power / float((shape * vol).sum()) * shape
    if solids:
        x, y, theta, info = solver_wall.solve(nx, common.bi_dict(Bi), common.theta_inf(known), solids=solids, Q=Q)
    else:
        x, y, theta, info = solver.solve(nx, common.bi_dict(Bi), common.theta_inf(known), Q=Q)
    if sink_fraction is not None:
        info["heater_power"] = heater_power
        info["sink_power"] = heater_power - info["q_in"]
    return x, y, theta, info


def _fv_volumes(x, y):
    h = x[1] - x[0]
    wx = np.full(len(x), h); wx[[0, -1]] = h / 2
    wy = np.full(len(y), h); wy[[0, -1]] = h / 2
    return np.outer(wx, wy)


def sample(x, y, theta, xy):
    """Bilinear interpolation at points xy (n, 2) of a field theta[ix, iy]."""
    return solver_wall.sample(x, y, theta, np.asarray(xy, float))  # refuses stencils touching a solid


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
    With a sink (info from solve_field(..., sink_fraction=f)), shares are fractions of the
    heater power, share_sink is listed separately, and the balance checks walls = heater - sink.
    """
    F = {w: float(info["total_flux"][w]) for w in common.ROBIN_WALLS}
    net = float(info["q_in"])  # heater minus sink (just the heater without a sink)
    H = float(info.get("heater_power", net))
    out = {w: max(F[w], 0.0) for w in F}
    out_sum = sum(out.values())
    total = sum(F.values()) + float(info["total_flux"]["W"])
    flows = {
        "flux": F,
        "source_integral": H,
        "share_signed": {w: F[w] / H for w in F},
        "share_outflow": {w: (out[w] / out_sum if out_sum > 0 else None) for w in F},
        "inflow": {w: min(F[w], 0.0) for w in F},
        "energy_balance_relerr": abs(total - net) / max(abs(H), 1e-12),
        "ranking": sorted(F, key=lambda w: -F[w]),
    }
    if "sink_power" in info:
        flows["share_sink"] = float(info["sink_power"]) / H
    return flows
