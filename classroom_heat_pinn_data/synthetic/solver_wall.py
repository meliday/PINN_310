"""Steady 2D heat-conduction solver with internal adiabatic solid walls (X1, LOG-073).

Same model and vertex-centred finite volumes as solver.py (k = 1), generalised with a cell
mask: a grid cell (square between four nodes) inside a solid rectangle is removed. A node's
control volume is the sum of the quarter-cells of its fluid cells, a face between two nodes
gets h/2 of length from each adjacent fluid cell, and an outer-wall node gets h/2 of Robin
boundary from each fluid cell touching that wall. Nothing couples across a solid cell, so the
solid surfaces are exactly no-flux; nodes with no fluid cell are not unknowns (theta = NaN).
With no solids the equations are those of solver.solve. Exactly conservative.

solids: list of dicts {"x0", "x1", "y0", "y1"}; edges should lie on grid lines.
"""
import numpy as np
from scipy import ndimage
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as spla

try:
    from . import solver
except ImportError:
    import solver

WALLS = solver.WALLS
LX, LY = solver.LX, solver.LY


def fluid_cells(x, y, solids):
    """(nx-1, ny-1) array, 1.0 for a fluid cell, 0.0 for a cell inside a solid."""
    xc, yc = (x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2
    XC, YC = np.meshgrid(xc, yc, indexing="ij")
    solid = np.zeros(XC.shape, dtype=bool)
    for s in solids or []:
        solid |= (XC > s["x0"]) & (XC < s["x1"]) & (YC > s["y0"]) & (YC < s["y1"])
    return (~solid).astype(float)


def _padded_cells_and_volumes(x, y, solids):
    """F[i+1, j+1] = fluid cell (i, j) with a zero border ("outside"), and node control volumes."""
    h = x[1] - x[0]
    F = np.zeros((len(x) + 1, len(y) + 1))
    F[1:-1, 1:-1] = fluid_cells(x, y, solids)
    return F, h * h / 4 * (F[:-1, :-1] + F[1:, :-1] + F[:-1, 1:] + F[1:, 1:])


def fluid_volumes(x, y, solids):
    """Fluid control volume of each node (0 inside solids); equals the base FV volumes without solids."""
    return _padded_cells_and_volumes(x, y, solids)[1]


def check_aligned(x, y, solids, tol=1e-9):
    h = x[1] - x[0]
    for s in solids or []:
        if not (s["x0"] < s["x1"] and s["y0"] < s["y1"]):
            raise ValueError(f"solid {s} is reversed or has zero area")
        for v in (s["x0"], s["x1"], s["y0"], s["y1"]):
            if abs(v / h - round(v / h)) > tol:
                raise ValueError(f"solid edge {v} is not on the grid (h={h})")


def check_topology(fluid, adjacency, robin_weight):
    """Reject geometries the vertex-centred scheme cannot represent (LOG-075 findings).

    fluid: (nx-1, ny-1) cell mask; adjacency: node graph (active nodes) of nonzero conductances;
    robin_weight: per active node, sum of Bi * boundary length.
    - X1-POINT-CONTACT-01: fluid that touches only at a corner shares a node, so that node's
      single unknown would join two pieces of fluid. Rejected locally (a node whose 2 x 2 cells
      are fluid on one diagonal only; LOG-080 showed a pinch inside an otherwise connected room
      keeps the region counts equal) and globally (cell and node region counts must agree).
    - X1-UNANCHORED-01: every connected fluid region needs a Robin boundary; an all-adiabatic
      region with a source has no steady solution.
    """
    P = np.pad(fluid > 0, 1)  # node (i, j) touches cells P[i, j], P[i+1, j], P[i, j+1], P[i+1, j+1]
    a, b, c, d = P[:-1, :-1], P[1:, :-1], P[:-1, 1:], P[1:, 1:]
    diagonal = (a & d & ~b & ~c) | (b & c & ~a & ~d)
    if diagonal.any():
        i, j = np.argwhere(diagonal)[0]
        raise ValueError(f"fluid touches only at a corner at node ({i}, {j}) "
                         f"({int(diagonal.sum())} such node(s)); point contacts are not supported")
    _, n_cells = ndimage.label(fluid > 0)  # face (4-) connectivity of fluid cells
    n_nodes, labels = csgraph.connected_components(adjacency, directed=False)
    if n_nodes != n_cells:
        raise ValueError(f"fluid regions touch only at a corner ({n_cells} cell regions, "
                         f"{n_nodes} node regions); point contacts are not supported")
    anchored = np.bincount(labels, weights=robin_weight, minlength=n_nodes) > 0
    if not anchored.all():
        raise ValueError(f"{int((~anchored).sum())} fluid region(s) have no Robin boundary (Bi > 0); "
                         "the steady problem has no unique solution")


def solve(nx, Bi, theta_inf, solids=None, Q=None, heaters=None, g=None):
    """Like solver.solve (k = 1) with solid rectangles removed.

    Returns x, y, theta (nx, ny; NaN at nodes inside solids), info. info["active"] marks the
    unknown nodes, info["vol"] the fluid control volumes, info["fluid_cells"] the cell mask.
    """
    x, y, h = solver.make_grid(nx)
    ny = len(y)
    check_aligned(x, y, solids)
    X, Y = np.meshgrid(x, y, indexing="ij")
    if Q is None:
        Q = solver.heater_source(X, Y, heaters or [])
    Bi = {w: Bi.get(w, 0.0) for w in WALLS}
    tinf = {w: theta_inf.get(w, 0.0) for w in WALLS}
    g = g or {}

    # F[i+1, j+1] is cell (i, j); the zero border stands for "outside the domain".
    F, vol = _padded_cells_and_volumes(x, y, solids)
    cx = 0.5 * (F[1:nx, :ny] + F[1:nx, 1:])      # face (i,j)-(i+1,j): length / h
    cy = 0.5 * (F[:nx, 1:ny] + F[1:, 1:ny])      # face (i,j)-(i,j+1)
    length = {"S": h / 2 * (F[:-1, 1] + F[1:, 1]), "N": h / 2 * (F[:-1, ny - 1] + F[1:, ny - 1]),
              "W": h / 2 * (F[1, :-1] + F[1, 1:]), "E": h / 2 * (F[nx - 1, :-1] + F[nx - 1, 1:])}
    segs = {"S": (np.s_[:, 0], x), "N": (np.s_[:, -1], x), "W": (np.s_[0, :], y), "E": (np.s_[-1, :], y)}

    active = vol > 0
    idx = np.full((nx, ny), -1)
    idx[active] = np.arange(active.sum())
    diag = np.zeros((nx, ny))
    rhs = Q * vol
    rows, cols, vals = [], [], []

    def add_pair(a, b, c):
        keep = c > 0
        a, b, c = a[keep], b[keep], c[keep]
        rows.extend([a, b]); cols.extend([b, a]); vals.extend([-c, -c])

    add_pair(idx[:-1, :], idx[1:, :], cx)
    diag[:-1, :] += cx; diag[1:, :] += cx
    add_pair(idx[:, :-1], idx[:, 1:], cy)
    diag[:, :-1] += cy; diag[:, 1:] += cy

    n = int(active.sum())
    off = sp.csr_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(n, n))
    robin = np.zeros((nx, ny))  # Bi * boundary length per node
    for w, (sl, _) in segs.items():
        robin[sl] += Bi[w] * length[w]
    check_topology(F[1:-1, 1:-1], off, robin[active])
    diag += robin

    boundary_forcing = {w: 0.0 for w in WALLS}
    for w, (sl, coord) in segs.items():
        rhs[sl] += Bi[w] * tinf[w] * length[w]
        if w in g:
            forcing = g[w](coord) * length[w]
            rhs[sl] -= forcing
            boundary_forcing[w] = float(forcing.sum())

    rows.append(idx[active]); cols.append(idx[active]); vals.append(diag[active])
    M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    theta = np.full((nx, ny), np.nan)
    theta[active] = spla.spsolve(M.tocsc(), rhs[active])

    q_in = float((Q * vol).sum())
    loss = {w: float((Bi[w] * (theta[sl] - tinf[w]) * length[w])[length[w] > 0].sum())
            for w, (sl, _) in segs.items()}
    total_flux = {w: loss[w] + boundary_forcing[w] for w in WALLS}
    balance_abs_err = abs(q_in - sum(total_flux.values()))
    info = {"h": h, "q_in": q_in, "loss": loss, "boundary_forcing": boundary_forcing,
            "total_flux": total_flux, "balance_abs_err": balance_abs_err,
            "balance_rel_err": balance_abs_err / max(abs(q_in), 1e-12), "Q": Q,
            "active": active, "vol": vol, "fluid_cells": F[1:-1, 1:-1], "matrix": M, "index": idx}
    return x, y, theta, info


def sample(x, y, field, pts):
    """Bilinear interpolation that refuses points whose stencil touches a solid (NaN) node.

    solver.sample would silently return NaN there (e.g. on a wall surface); a fluid-aware
    interpolation is not implemented, so such points must be grid nodes or kept off the wall.
    """
    values = solver.sample(x, y, field, pts)
    if not np.all(np.isfinite(values)):
        bad = np.asarray(pts)[~np.isfinite(values)]
        raise ValueError(f"sample points next to a solid cannot be interpolated: {bad.tolist()}")
    return values


def mms_wall_test(solid, nxs=(51, 101, 201, 401), a=0.3, b=20.0):
    """Manufactured solution with zero normal derivative on every face of `solid`.

    theta = 1 + a cos(pi x / x1) + b [(y - yc)^3 / 3 - d^2 (y - yc)], with yc the solid's mid
    height and d its half thickness: d theta/dy = b [(y - yc)^2 - d^2] = 0 on y0 and y1, and
    d theta/dx = 0 on x = x1 (and on the adiabatic W wall x = 0). Requires solid x0 = 0.
    Returns grid spacings, max nodal errors and observed orders.
    """
    x1, yc, d = solid["x1"], (solid["y0"] + solid["y1"]) / 2, (solid["y1"] - solid["y0"]) / 2
    p = np.pi / x1
    th = lambda X, Y: 1 + a * np.cos(p * X) + b * ((Y - yc) ** 3 / 3 - d * d * (Y - yc))
    thx = lambda X, Y: -a * p * np.sin(p * X) + 0 * Y
    thy = lambda X, Y: b * ((Y - yc) ** 2 - d * d) + 0 * X
    Qm = lambda X, Y: a * p * p * np.cos(p * X) - 2 * b * (Y - yc)
    Bi = {"S": 5.0, "E": 1.0, "N": 0.5, "W": 0.0}
    tinf = {"S": 0.0, "E": 0.0, "N": 0.9, "W": 0.0}
    g = {"S": lambda s: thy(s, 0 * s) - Bi["S"] * (th(s, 0 * s) - tinf["S"]),
         "N": lambda s: -thy(s, LY + 0 * s) - Bi["N"] * (th(s, LY + 0 * s) - tinf["N"]),
         "E": lambda s: -thx(LX + 0 * s, s) - Bi["E"] * (th(LX + 0 * s, s) - tinf["E"])}
    hs, errs = [], []
    for nx in nxs:
        x, y, h = solver.make_grid(nx)
        X, Y = np.meshgrid(x, y, indexing="ij")
        _, _, T, info = solve(nx, Bi, tinf, solids=[solid], Q=Qm(X, Y), g=g)
        act = info["active"]
        errs.append(float(np.abs(T[act] - th(X, Y)[act]).max())); hs.append(h)
    orders = [float(np.log(errs[i] / errs[i + 1]) / np.log(hs[i] / hs[i + 1])) for i in range(len(errs) - 1)]
    return hs, errs, orders
