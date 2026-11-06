"""Steady 2D heat-conduction solver for a rectangular classroom (nondimensional).

Model
    -div(k grad theta) = Q            in  [0, LX] x [0, LY]
    -k d(theta)/dn = Bi_i (theta - theta_inf_i) + g_i   on wall i

Walls: 'S' (y=0, window), 'E' (x=LX, exterior), 'N' (y=LY, corridor), 'W' (x=0, adiabatic by default).
Discretisation: vertex-centred finite volumes on a uniform grid (same spacing h in x and y).
With k = 1 this is identical to the 5-point Laplacian + ghost-point Robin scheme.
The scheme is exactly conservative, so discrete energy balance holds to round-off.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

LX, LY = 1.0, 0.8
WALLS = ("S", "E", "N", "W")


def make_grid(nx):
    """nx points in x; ny chosen so that the spacing is equal in x and y."""
    h = LX / (nx - 1)
    ny = int(round(LY / h)) + 1
    assert abs((ny - 1) * h - LY) < 1e-12, "LY must be a multiple of h"
    x = np.linspace(0, LX, nx)
    y = np.linspace(0, LY, ny)
    return x, y, h


def heater_source(X, Y, heaters):
    """heaters: list of dicts {x, y, A, s}; Gaussian blobs."""
    Q = np.zeros_like(X)
    for ht in heaters:
        Q += ht["A"] * np.exp(-((X - ht["x"]) ** 2 + (Y - ht["y"]) ** 2) / (2 * ht["s"] ** 2))
    return Q


def solve(nx, Bi, theta_inf, Q=None, heaters=None, k=None, g=None):
    """Solve the steady problem.

    Bi, theta_inf: dicts keyed by wall ('S','E','N','W'); missing -> 0 (adiabatic / 0).
    Q: array (nx, ny) or None (then built from heaters).
    k: callable k(X, Y) or None (k = 1).
    g: dict wall -> callable g(x_or_y) extra boundary flux (for MMS), or None.
    Returns x, y, theta (nx, ny), info dict.
    info['loss'] contains the Robin flux only, for backward compatibility.
    info['total_flux'] adds the prescribed g flux and is used for energy balance.
    """
    x, y, h = make_grid(nx)
    ny = len(y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    if Q is None:
        Q = heater_source(X, Y, heaters or [])
    kf = (lambda XX, YY: np.ones_like(XX)) if k is None else k
    Bi = {w: Bi.get(w, 0.0) for w in WALLS}
    tinf = {w: theta_inf.get(w, 0.0) for w in WALLS}
    g = g or {}

    # control-volume sizes: h in the interior, h/2 at boundaries
    wx = np.full(nx, h); wx[0] = wx[-1] = h / 2
    wy = np.full(ny, h); wy[0] = wy[-1] = h / 2
    vol = np.outer(wx, wy)

    idx = np.arange(nx * ny).reshape(nx, ny)
    rows, cols, vals = [], [], []
    diag = np.zeros((nx, ny))
    rhs = Q * vol

    # x-direction faces between (i,j) and (i+1,j): face length wy[j], conductance k_face*wy/h
    kx_face = kf((X[:-1, :] + X[1:, :]) / 2, Y[:-1, :])
    cx = kx_face * wy[None, :] / h
    # y-direction faces between (i,j) and (i,j+1)
    ky_face = kf(X[:, :-1], (Y[:, :-1] + Y[:, 1:]) / 2)
    cy = ky_face * wx[:, None] / h

    def add_pair(a, b, c):
        rows.extend([a.ravel(), b.ravel()]); cols.extend([b.ravel(), a.ravel()])
        vals.extend([-c.ravel(), -c.ravel()])

    add_pair(idx[:-1, :], idx[1:, :], cx)
    diag[:-1, :] += cx; diag[1:, :] += cx
    add_pair(idx[:, :-1], idx[:, 1:], cy)
    diag[:, :-1] += cy; diag[:, 1:] += cy

    # boundary Robin fluxes: outflow = Bi*(theta - tinf)*length + g*length
    segs = {"S": (np.s_[:, 0], wx, x), "N": (np.s_[:, -1], wx, x),
            "W": (np.s_[0, :], wy, y), "E": (np.s_[-1, :], wy, y)}
    boundary_forcing = {w: 0.0 for w in WALLS}
    for w, (sl, length, coord) in segs.items():
        diag[sl] += Bi[w] * length
        rhs[sl] += Bi[w] * tinf[w] * length
        if w in g:
            forcing = g[w](coord) * length
            rhs[sl] -= forcing
            boundary_forcing[w] = float(forcing.sum())

    rows.append(idx.ravel()); cols.append(idx.ravel()); vals.append(diag.ravel())
    M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nx * ny, nx * ny))
    theta = spla.spsolve(M.tocsc(), rhs.ravel()).reshape(nx, ny)

    # energy balance (discrete, using the same control volumes / boundary lengths)
    q_in = float((Q * vol).sum())
    loss = {}
    for w, (sl, length, _) in segs.items():
        loss[w] = float((Bi[w] * (theta[sl] - tinf[w]) * length).sum())
    total_flux = {w: loss[w] + boundary_forcing[w] for w in WALLS}
    balance_abs_err = abs(q_in - sum(total_flux.values()))
    info = {"h": h, "q_in": q_in, "loss": loss,
            "boundary_forcing": boundary_forcing, "total_flux": total_flux,
            "balance_abs_err": balance_abs_err,
            "balance_rel_err": balance_abs_err / max(abs(q_in), 1e-12),
            "Q": Q}
    return x, y, theta, info


def sample(x, y, field, pts):
    """Bilinear interpolation of a grid field at points (n,2)."""
    from scipy.interpolate import RegularGridInterpolator
    return RegularGridInterpolator((x, y), field)(pts)


# ---------------------------------------------------------------- verification
def mms_test(verbose=True):
    """Manufactured solution with variable k, Robin walls and a Neumann W wall."""
    a, b, c = 0.3, 0.1, 0.3
    p, q = np.pi, np.pi / LY
    th = lambda X, Y: 1 + a * np.cos(p * X) * np.cos(q * Y) + b * X
    thx = lambda X, Y: -a * p * np.sin(p * X) * np.cos(q * Y) + b
    thy = lambda X, Y: -a * q * np.cos(p * X) * np.sin(q * Y)
    k = lambda X, Y: 1 + c * np.sin(p * X) * np.sin(q * Y)
    kx = lambda X, Y: c * p * np.cos(p * X) * np.sin(q * Y)
    ky = lambda X, Y: c * q * np.sin(p * X) * np.cos(q * Y)

    def Qm(X, Y):
        lap = -a * (p ** 2 + q ** 2) * np.cos(p * X) * np.cos(q * Y)
        return -(k(X, Y) * lap + kx(X, Y) * thx(X, Y) + ky(X, Y) * thy(X, Y))

    Bi = {"S": 5.0, "E": 1.0, "N": 0.5, "W": 0.0}
    tinf = {"S": 0.0, "E": 0.0, "N": 0.9, "W": 0.0}
    # outward normal derivatives: S: -d/dy, N: +d/dy, W: -d/dx, E: +d/dx
    g = {
        "S": lambda s: -k(s, 0 * s) * (-thy(s, 0 * s)) - Bi["S"] * (th(s, 0 * s) - tinf["S"]),
        "N": lambda s: -k(s, LY + 0 * s) * (thy(s, LY + 0 * s)) - Bi["N"] * (th(s, LY + 0 * s) - tinf["N"]),
        "W": lambda s: -k(0 * s, s) * (-thx(0 * s, s)) - Bi["W"] * (th(0 * s, s) - tinf["W"]),
        "E": lambda s: -k(LX + 0 * s, s) * (thx(LX + 0 * s, s)) - Bi["E"] * (th(LX + 0 * s, s) - tinf["E"]),
    }
    errs, hs = [], []
    for nx in (26, 51, 101, 201):
        x, y, h = make_grid(nx)
        X, Y = np.meshgrid(x, y, indexing="ij")
        _, _, T, info = solve(nx, Bi, tinf, Q=Qm(X, Y), k=k, g=g)
        errs.append(np.abs(T - th(X, Y)).max()); hs.append(h)
    orders = [np.log(errs[i] / errs[i + 1]) / np.log(hs[i] / hs[i + 1]) for i in range(len(errs) - 1)]
    if verbose:
        for h_, e in zip(hs, errs):
            print(f"  MMS h={h_:.4f}  max error={e:.3e}")
        print("  observed orders:", np.round(orders, 3))
    return hs, errs, orders


if __name__ == "__main__":
    print("MMS test (variable k, Robin and Neumann boundary forcing):")
    mms_test()
