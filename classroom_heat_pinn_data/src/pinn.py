"""PINN for the steady 2D classroom model (k = 1).

    r_Omega = -(theta_xx + theta_yy) - Q
    r_i     = grad(theta) . n_i + Bi_i (theta - theta_inf_i)   (S, E, N)
    r_W     = grad(theta) . n_W                                 (adiabatic)

Network input is transformed inside forward(); derivatives are taken by autograd
with respect to the raw (x, y). Bi = exp(beta). Forward mode (E1) fixes Bi as a
buffer; inverse mode (E2) learns beta from a fresh network with beta = 0.
"""
import time

import numpy as np
import torch
from torch import nn

try:
    from . import common
except ImportError:
    import src.common as common

DTYPE = torch.float64
# Outward unit normals; S/N are horizontal walls (y = 0, LY), W/E vertical (x = 0, LX).
NORMALS = {"S": (0.0, -1.0), "E": (1.0, 0.0), "N": (0.0, 1.0), "W": (-1.0, 0.0)}
WALL_ORDER = ("S", "E", "N", "W")

DEFAULT_EXPERIMENT = {
    "seed": 0, "hidden": 64, "depth": 4,
    "n_interior": 1000, "n_wall": 100, "heater_frac": 0.3,
    "lr_net": 1e-3, "lr_beta": 3e-3, "adam_steps": 5000, "resample_every": 500,
    "lbfgs_iters": 0, "lambda_pde": 1.0, "lambda_bc": 1.0, "lambda_data": 1.0,
    "log_every": 100, "time_budget_s": None, "threads": None,
}


class MLP(nn.Module):
    """tanh MLP with Xavier init; output 0.9 + net(x~, y~), no clipping."""

    def __init__(self, LX, LY, hidden=64, depth=4, offset=0.9):
        super().__init__()
        layers, d = [], 2
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.body = nn.Sequential(*layers)
        for m in self.body:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        self.LX, self.LY, self.offset = LX, LY, offset

    def forward(self, xy):
        z = torch.stack([2 * xy[:, 0] / self.LX - 1, 2 * xy[:, 1] / self.LY - 1], dim=1)
        return self.offset + self.body(z).squeeze(1)


class HeatPINN(nn.Module):
    def __init__(self, known, hidden=64, depth=4, bi_fixed=None):
        super().__init__()
        self.LX, self.LY = known["domain"]["LX"], known["domain"]["LY"]
        self.net = MLP(self.LX, self.LY, hidden, depth).to(DTYPE)
        tinf = common.theta_inf(known)
        self.register_buffer("theta_inf", torch.tensor([tinf[w] for w in common.ROBIN_WALLS], dtype=DTYPE))
        if bi_fixed is None:
            self.beta = nn.Parameter(torch.zeros(3, dtype=DTYPE))  # Bi = 1 at start
            self.inverse = True
        else:
            bi = common.bi_dict(bi_fixed)
            self.register_buffer("beta", torch.log(torch.tensor([bi[w] for w in common.ROBIN_WALLS],
                                                                dtype=DTYPE)))
            self.inverse = False

    def bi(self):
        return torch.exp(self.beta)

    def forward(self, xy):
        return self.net(xy)


# ------------------------------------------------------------------ physics
def source(xy, known, key="heater"):
    h = known[key]
    r2 = (xy[:, 0] - h["x"]) ** 2 + (xy[:, 1] - h["y"]) ** 2
    return h["A"] * torch.exp(-r2 / (2 * h["s"] ** 2))


def derivatives(fn, xy, second=True):
    """theta, grad (n, 2) and Laplacian (n,) of fn at xy by autograd (raw coordinates)."""
    xy = xy.detach().clone().requires_grad_(True)
    theta = fn(xy)
    grad = torch.autograd.grad(theta.sum(), xy, create_graph=True)[0]
    if not second:
        return theta, grad, None
    lap = 0.0
    for k in range(2):
        lap = lap + torch.autograd.grad(grad[:, k].sum(), xy, create_graph=True)[0][:, k]
    return theta, grad, lap


def pde_residual(model, xy, known):
    _, _, lap = derivatives(model, xy)
    return -lap - source(xy, known)


def bc_residuals(model, wall_pts):
    """{wall: residual} with r = grad.n + Bi (theta - theta_inf); W is grad.n."""
    bi = model.bi()
    out = {}
    for w, xy in wall_pts.items():
        theta, grad, _ = derivatives(model, xy, second=False)
        n = torch.tensor(NORMALS[w], dtype=DTYPE)
        r = grad @ n
        if w != "W":
            i = common.ROBIN_WALLS.index(w)
            r = r + bi[i] * (theta - model.theta_inf[i])
        out[w] = r
    return out


# ------------------------------------------------------------------ sampling
def sample_interior(n, known, gen, heater_frac=0.3):
    if not 0.0 <= heater_frac <= 1.0:
        raise ValueError(f"heater_frac must be in [0, 1], got {heater_frac}")
    LX, LY = known["domain"]["LX"], known["domain"]["LY"]
    n_h = int(round(heater_frac * n))
    uni = torch.rand(n - n_h, 2, generator=gen, dtype=DTYPE) * torch.tensor([LX, LY], dtype=DTYPE)
    if n_h == 0:
        return uni
    h = known["heater"]
    pts = []
    while sum(len(p) for p in pts) < n_h:  # Gaussian around the heater, rejected outside
        c = torch.randn(2 * n_h, 2, generator=gen, dtype=DTYPE) * h["s"] + torch.tensor([h["x"], h["y"]], dtype=DTYPE)
        keep = (c[:, 0] > 0) & (c[:, 0] < LX) & (c[:, 1] > 0) & (c[:, 1] < LY)
        pts.append(c[keep])
    near = torch.cat(pts)[:n_h]
    return torch.cat([uni, near])


def sample_walls(n, known, gen):
    """n points per wall on the open edge (corners excluded, one normal per point)."""
    LX, LY = known["domain"]["LX"], known["domain"]["LY"]
    eps = 1e-6
    s = lambda L: eps + (L - 2 * eps) * torch.rand(n, generator=gen, dtype=DTYPE)
    zeros, ones = torch.zeros(n, dtype=DTYPE), torch.ones(n, dtype=DTYPE)
    return {
        "S": torch.stack([s(LX), zeros], 1), "N": torch.stack([s(LX), LY * ones], 1),
        "W": torch.stack([zeros, s(LY)], 1), "E": torch.stack([LX * ones, s(LY)], 1),
    }


# ------------------------------------------------------------------ loss
def losses(model, known, interior, walls, obs_xy=None, obs_theta=None, exp=DEFAULT_EXPERIMENT):
    r_pde = pde_residual(model, interior, known)
    r_bc = bc_residuals(model, walls)
    mse_pde = (r_pde ** 2).mean()
    mse_wall = {w: (r ** 2).mean() for w, r in r_bc.items()}
    mse_bc = sum(mse_wall.values()) / len(mse_wall)  # mean per wall, then over walls
    if obs_xy is not None:
        mse_data = ((model(obs_xy) - obs_theta) ** 2).mean()
    else:
        mse_data = torch.zeros((), dtype=DTYPE)
    total = exp["lambda_data"] * mse_data + exp["lambda_pde"] * mse_pde + exp["lambda_bc"] * mse_bc
    return total, {"data": mse_data, "pde": mse_pde, "bc": mse_bc,
                   **{f"bc_{w}": v for w, v in mse_wall.items()}}


# ------------------------------------------------------------------ diagnostics
def _trapz(f, x):
    return float(np.sum((f[1:] + f[:-1]) * np.diff(x)) / 2)


def heat_flows(model, known, n=401):
    """Signed wall flows of the PINN field, outflow positive (same convention as fd.heat_flows).

    flux:      Robin flux  int Bi (theta - theta_inf) ds  (what the BC says leaves)
    flux_grad: gradient flux int -grad(theta).n ds (what the network field carries)
    The two differ where the BC residual is not satisfied. W uses only flux_grad.
    """
    LX, LY = known["domain"]["LX"], known["domain"]["LY"]
    sx, sy = np.linspace(0, LX, n), np.linspace(0, LY, n)
    pts = {"S": np.c_[sx, 0 * sx], "N": np.c_[sx, LY + 0 * sx],
           "W": np.c_[0 * sy, sy], "E": np.c_[LX + 0 * sy, sy]}
    bi = model.bi().detach().numpy()
    tinf = model.theta_inf.numpy()
    flux, flux_grad = {}, {}
    for w, p in pts.items():
        theta, grad, _ = derivatives(model, torch.tensor(p, dtype=DTYPE), second=False)
        theta, grad = theta.detach().numpy(), grad.detach().numpy()
        coord = p[:, 0] if w in ("S", "N") else p[:, 1]
        flux_grad[w] = _trapz(-(grad @ np.array(NORMALS[w])), coord)
        if w != "W":
            i = common.ROBIN_WALLS.index(w)
            flux[w] = _trapz(bi[i] * (theta - tinf[i]), coord)
    # Source integral by the same trapezoid rule on a fine tensor grid.
    gx, gy = np.linspace(0, LX, 401), np.linspace(0, LY, 321)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    Q = source(torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=DTYPE), known).numpy().reshape(X.shape)
    H = _trapz(np.array([_trapz(Q[i], gy) for i in range(len(gx))]), gx)
    out = {w: max(flux[w], 0.0) for w in flux}
    out_sum = sum(out.values())
    return {
        "flux": flux, "flux_grad": flux_grad, "source_integral": H,
        "share_signed": {w: flux[w] / H for w in flux},
        "share_outflow": {w: (out[w] / out_sum if out_sum > 0 else None) for w in flux},
        "inflow": {w: min(flux[w], 0.0) for w in flux},
        # Robin-flux balance: zero only if PDE and all BCs (incl. adiabatic W) hold.
        "energy_balance_relerr": abs(sum(flux.values()) - H) / max(abs(H), 1e-12),
        # Gradient-flux balance over all four walls: checks the PDE (divergence theorem).
        "gradient_balance_relerr": abs(sum(flux_grad.values()) - H) / max(abs(H), 1e-12),
        "ranking": sorted(flux, key=lambda w: -flux[w]),
    }


def predict_grid(model, x, y, batch=20000):
    """theta[ix, iy] on a tensor grid (indexing='ij')."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    pts = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=DTYPE)
    with torch.no_grad():
        out = torch.cat([model(pts[i:i + batch]) for i in range(0, len(pts), batch)])
    return out.numpy().reshape(X.shape)


# ------------------------------------------------------------------ training
class _BudgetExceeded(Exception):
    pass


def _finite(*tensors):
    return all(bool(torch.isfinite(t).all()) for t in tensors)


def train(model, known, experiment=None, obs=None, on_log=None):
    """Adam (+ optional L-BFGS) on a fixed experiment config.

    obs: common.Observations or None (pure forward solve). Truth is never passed in.
    Returns a history list and a status dict; stops early (status 'failed') on NaN/Inf.
    """
    exp = {**DEFAULT_EXPERIMENT, **(experiment or {})}
    if exp["threads"]:
        torch.set_num_threads(exp["threads"])
    torch.manual_seed(exp["seed"])
    gen = torch.Generator().manual_seed(exp["seed"])
    obs_xy = obs_theta = None
    if obs is not None:
        obs_xy = torch.tensor(obs.xy, dtype=DTYPE)
        obs_theta = torch.tensor(obs.theta, dtype=DTYPE)

    groups = [{"params": list(model.net.parameters()), "lr": exp["lr_net"]}]
    if model.inverse:
        groups.append({"params": [model.beta], "lr": exp["lr_beta"]})
    opt = torch.optim.Adam(groups)

    # status["stage"] is the last optimizer that ran ("adam" or "lbfgs").
    history, status = [], {"status": "completed", "stage": "adam", "reason": None}
    t0 = time.perf_counter()
    interior = walls = None
    budget = exp["time_budget_s"]

    def over_budget():
        return bool(budget) and time.perf_counter() - t0 > budget

    def log(step, stage, total, parts):
        g = model.beta.grad if model.inverse and model.beta.grad is not None else None
        rec = {"step": step, "stage": stage, "elapsed_s": time.perf_counter() - t0,
               "loss": float(total), **{k: float(v) for k, v in parts.items()},
               "Bi": dict(zip(common.ROBIN_WALLS, model.bi().detach().tolist())),
               "beta_grad_norm": float(g.norm()) if g is not None else None}
        history.append(rec)
        if on_log:
            on_log(rec)

    step = 0
    for step in range(exp["adam_steps"]):
        if step % exp["resample_every"] == 0:
            interior = sample_interior(exp["n_interior"], known, gen, exp["heater_frac"])
            walls = sample_walls(exp["n_wall"], known, gen)
        opt.zero_grad()
        total, parts = losses(model, known, interior, walls, obs_xy, obs_theta, exp)
        total.backward()
        params_ok = _finite(*[p for grp in groups for p in grp["params"]])
        grads_ok = _finite(*[p.grad for grp in groups for p in grp["params"] if p.grad is not None])
        if not (torch.isfinite(total) and params_ok and grads_ok):
            status = {"status": "failed", "stage": "adam", "step": step,
                      "reason": "non-finite loss, parameter or gradient"}
            log(step, "adam", total.detach(), {k: v.detach() for k, v in parts.items()})
            return history, status
        if step % exp["log_every"] == 0:
            log(step, "adam", total.detach(), {k: v.detach() for k, v in parts.items()})
        opt.step()
        if over_budget():
            status.update(reason=f"adam stopped by time budget at step {step + 1}")
            break
    status["adam_steps_done"] = step + 1
    params = [p for grp in groups for p in grp["params"]]

    # The time budget covers the whole run: L-BFGS starts only if budget remains and
    # its closure re-checks it. An aborted L-BFGS restores the pre-L-BFGS parameters,
    # since a line search can stop at an unevaluated trial point.
    if exp["lbfgs_iters"] and over_budget():
        status["lbfgs"] = "skipped: time budget exhausted"
    elif exp["lbfgs_iters"]:
        status["stage"] = "lbfgs"
        saved = [p.detach().clone() for p in params]
        lbfgs = torch.optim.LBFGS(params, lr=1.0, max_iter=exp["lbfgs_iters"],
                                  line_search_fn="strong_wolfe")

        def closure():  # collocation fixed during L-BFGS -> deterministic closure
            if over_budget():
                raise _BudgetExceeded
            lbfgs.zero_grad()
            tot, _ = losses(model, known, interior, walls, obs_xy, obs_theta, exp)
            tot.backward()
            return tot

        try:
            lbfgs.step(closure)
            status["lbfgs"] = "completed"
        except _BudgetExceeded:
            with torch.no_grad():
                for p, v in zip(params, saved):
                    p.copy_(v)
            status["lbfgs"] = "aborted: time budget exhausted; parameters restored to pre-L-BFGS"
            status["reason"] = "L-BFGS stopped by time budget"

    # Final check after the last update of either optimizer.
    total, parts = losses(model, known, interior, walls, obs_xy, obs_theta, exp)
    final_stage = "lbfgs" if status.get("lbfgs") == "completed" else "final"
    log(step + 1, final_stage, total.detach(), {k: v.detach() for k, v in parts.items()})
    if not (torch.isfinite(total) and _finite(*params)):
        status.update(status="failed", reason=f"non-finite loss or parameter after {status['stage']}")

    status["wall_time_s"] = time.perf_counter() - t0
    status["device"] = "cpu"
    status["threads"] = torch.get_num_threads()
    status["dtype"] = str(DTYPE)
    return history, status


def save_checkpoint(path, model, experiment, history, status, extra=None):
    torch.save({"state_dict": model.state_dict(), "inverse": model.inverse,
                "experiment": {**DEFAULT_EXPERIMENT, **(experiment or {})},
                "history": history, "status": status, "extra": extra or {}}, path)


def count_parameters(model):
    return sum(p.numel() for p in model.net.parameters())
