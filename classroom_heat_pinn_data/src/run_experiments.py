"""Experiment entry point.

    cd classroom_heat_pinn_data && python -m src.run_experiments --exp E0
    cd classroom_heat_pinn_data && python -m src.run_experiments --exp E1 --seed 0

Fitting (RBF, FD-inverse) receives only known.json and sensor observations.
Truth is loaded afterwards through evaluate.py and every access is logged.
E1 is the exception: a diagnostic that fixes the true Bi in a forward PINN.
"""
import argparse
import csv
import json
import platform
import sys
import time

import numpy as np

try:
    from ..synthetic import solver
    from . import baselines, common, evaluate, fd
except ImportError:
    from synthetic import solver
    import src.baselines as baselines
    import src.common as common
    import src.evaluate as evaluate
    import src.fd as fd


def environment():
    import scipy
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "machine": platform.machine()}


METRIC_KEY = ("exp_id", "method", "data_file", "seed")


def _metric_key(row):
    return tuple("" if row.get(k) is None else str(row[k]) for k in METRIC_KEY)


def write_metrics(rows, path=None):
    """Upsert rows into results/metrics.csv by (exp_id, method, data_file, seed)."""
    path = path or common.RESULTS_DIR / "metrics.csv"
    new_keys = {_metric_key(r) for r in rows}
    kept = []
    if path.exists():
        with open(path, newline="") as fh:
            kept = [r for r in csv.DictReader(fh) if _metric_key(r) not in new_keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=evaluate.METRIC_COLUMNS)
        w.writeheader()
        for r in kept + rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in evaluate.METRIC_COLUMNS})


def tool_checks(known, obs):
    """Solver, grid, unit and orientation checks. Uses truth: diagnostic only.

    obs must be a clean observation file: its values are compared to the reference.
    """
    purpose = "E0 tool diagnostic (solver reproduction / orientation)"
    out = {}
    mms_h, mms_errs, orders = solver.mms_test(verbose=False)
    out["mms_h"] = mms_h
    out["mms_max_error"] = mms_errs
    out["mms_observed_order"] = orders

    bi_true = evaluate.load_truth_bi(purpose)
    xc, yc, thc, infoc = fd.solve_field(bi_true, known, nx=fd.INVERSE_NX)
    xf, yf, thf, infof = fd.solve_field(bi_true, known, nx=fd.FINE_NX)
    out["energy_relerr_101"] = infoc["balance_rel_err"]
    out["energy_relerr_201"] = infof["balance_rel_err"]
    out["refinement_maxdiff_C"] = float(np.abs(thf[::2, ::2] - thc).max() * common.dt_K(known))

    ref = evaluate.load_reference(purpose)
    same_axes = np.allclose(ref["x"], xf) and np.allclose(ref["y"], yf)
    out["reference_axes_match"] = bool(same_axes)
    out["reference_reproduction_maxdiff_C"] = float(np.abs(thf - ref["theta"]).max() * common.dt_K(known))
    out["reference_T_equals_DT_theta_maxdiff_C"] = float(
        np.abs(ref["T_C"] - common.dt_K(known) * ref["theta"]).max())

    # Orientation: bilinear sampling == node value == CSV observation (clean data).
    for name, (x, y, th) in {"101": (xc, yc, thc), "201": (xf, yf, thf)}.items():
        ix, iy = fd.node_indices(x, y, obs.xy)
        out[f"sample_vs_node_maxdiff_{name}"] = float(np.abs(fd.sample(x, y, th, obs.xy) - th[ix, iy]).max())
    ix, iy = fd.node_indices(xf, yf, obs.xy)
    out["reference_vs_csv_maxdiff_C"] = float(
        np.abs(ref["theta"][ix, iy] - obs.theta).max() * common.dt_K(known))
    # Swapping the sensor coordinates must disagree; guards against an [iy, ix] mix-up.
    in_box = obs.xy[:, 0] <= yf[-1]
    swapped = fd.sample(xf, yf, ref["theta"], obs.xy[in_box][:, ::-1])
    out["swapped_xy_maxdiff_C"] = float(np.abs(swapped - obs.theta[in_box]).max() * common.dt_K(known))

    out["passed"] = {
        "mms_order_gt_1.8": bool(min(orders) > 1.8),
        "energy_lt_1e-8": bool(max(out["energy_relerr_101"], out["energy_relerr_201"]) < 1e-8),
        "refinement_lt_0.01C": bool(out["refinement_maxdiff_C"] < 0.01),
        "reference_reproduced_lt_1e-6C": bool(same_axes and out["reference_reproduction_maxdiff_C"] < 1e-6),
        "csv_matches_reference_lt_1e-4C": bool(out["reference_vs_csv_maxdiff_C"] < 1e-4),
    }
    return out


def plot_e0(ref, fields, obs, flows_true, flows_fd, known, fig_dir, data_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dT = common.dt_K(known)
    extent = [0, known["domain"]["LX"], 0, known["domain"]["LY"]]
    T_ref = dT * ref["theta"]
    vmin, vmax = np.percentile(T_ref, [0.5, 99.5])
    errs = {k: dT * (v - ref["theta"]) for k, v in fields.items()}
    elim = max(np.abs(e).max() for e in errs.values())

    fig, axes = plt.subplots(2, 1 + len(fields), figsize=(4.2 * (1 + len(fields)), 6.4),
                             constrained_layout=True)
    panels = [("Reference (synthetic truth)", T_ref)] + [(k, dT * v) for k, v in fields.items()]
    for ax, (title, T) in zip(axes[0], panels):
        im = ax.imshow(T.T, origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap="inferno")
        ax.set_title(title)
    fig.colorbar(im, ax=axes[0], label="T [°C]")
    axes[1, 0].axis("off")
    axes[1, 0].text(0.0, 0.5, f"Simplified 2D model,\nsynthetic data (sensors_{data_name})\n"
                    "Error = prediction − reference\nSame colour limits per row",
                    va="center", fontsize=10)
    for ax, (title, E) in zip(axes[1, 1:], errs.items()):
        im2 = ax.imshow(E.T, origin="lower", extent=extent, vmin=-elim, vmax=elim, cmap="RdBu_r")
        ax.set_title(f"{title} error")
    fig.colorbar(im2, ax=axes[1, 1:], label="ΔT [°C]")
    for ax in axes.ravel():
        if ax.axison:
            ax.plot(obs.xy[:, 0], obs.xy[:, 1], "o", mfc="none", mec="cyan", ms=6)
            ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.savefig(fig_dir / f"e0_{data_name}_fields.png", dpi=130)
    plt.close(fig)

    walls = list(common.ROBIN_WALLS)
    labels = {"S": "S window", "E": "E exterior", "N": "N corridor"}
    xs = np.arange(len(walls))
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for off, (name, fl) in zip((-0.2, 0.2), (("Truth (201 grid)", flows_true), ("FD-inverse (101 grid)", flows_fd))):
        ax.bar(xs + off, [100 * fl["share_signed"][w] for w in walls], width=0.4, label=name)
    ax.set_xticks(xs, [labels[w] for w in walls])
    ax.set_ylabel("Signed wall flow / heater source [%]")
    ax.axhline(0, color="k", lw=0.8)
    ax.legend()
    ax.set_title(f"E0: wall heat flow (synthetic, {data_name})")
    fig.savefig(fig_dir / f"e0_{data_name}_heatflow.png", dpi=130)
    plt.close(fig)


def _write_summary(out_dir, summary):
    (out_dir / "summary.json").write_text(json.dumps(common.to_jsonable(summary), indent=2))


def run_e0(data_name="A_clean"):
    known = common.load_known()
    obs = common.load_observations(data_name, known)
    data_file = f"sensors_{data_name}.csv"
    out_dir = common.RESULTS_DIR / "e0" / data_name
    fig_dir = common.RESULTS_DIR / "figures"
    for d in (out_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)
    summary = {"exp_id": "E0", "data_file": data_file, "environment": environment(),
               "note": "Simplified 2D model, synthetic data. E0 is a tool/baseline check, "
                       "not a PINN result."}

    # ---- tool diagnostics first (truth access logged); stop if any check fails.
    # Compared against the clean file of the same sensor layout, since noisy or
    # mismatch observations are not expected to equal the reference.
    layout = data_name.split("_")[0]
    checks = tool_checks(known, common.load_observations(f"{layout}_clean", known))
    summary["tool_checks"] = checks
    failed = [k for k, ok in checks["passed"].items() if not ok]
    if failed:
        note = "not run: tool checks failed: " + ", ".join(failed)
        write_metrics([evaluate.metrics_row(exp_id="E0", method=m, data_file=data_file,
                                            status="failed", notes=note)
                       for m in ("RBF_tps", "FD_inverse")])
        summary.update(status="failed", reason=note)
        _write_summary(out_dir, summary)
        return summary

    # ---- fitting: observations + known only; fitting functions never receive truth
    xq, yq = np.linspace(0, 1, fd.FINE_NX), np.linspace(0, 0.8, 161)
    Xq, Yq = np.meshgrid(xq, yq, indexing="ij")
    query = np.c_[Xq.ravel(), Yq.ravel()]
    t0 = time.perf_counter()
    theta_rbf = baselines.reconstruct_rbf(obs.xy, obs.theta, query).reshape(Xq.shape)
    rbf_time = time.perf_counter() - t0

    fit = baselines.fit_fd_inverse(obs, known)
    (out_dir / "fd_inverse_runs.json").write_text(json.dumps(common.to_jsonable(fit["runs"]), indent=2))

    # ---- evaluation (truth access logged)
    ref = evaluate.load_reference("E0 evaluation: Q1 metrics")
    if not (np.allclose(ref["x"], xq) and np.allclose(ref["y"], yq)):
        raise ValueError("reference grid differs from the evaluation query grid")
    q1_cols = ("q1_rmse_C", "q1_maxabs_C", "q1_wall_rmse_C")
    q1_rbf = evaluate.q1_metrics(theta_rbf, ref, obs, known)
    rows = [evaluate.metrics_row(exp_id="E0", method="RBF_tps", data_file=data_file, status="completed",
                                 n_unknown=0, wall_time_s=rbf_time, **{k: q1_rbf[k] for k in q1_cols},
                                 notes="Q1 only; Bi/flux/what-if N/A")]
    summary["rbf"] = {"q1": q1_rbf, "wall_time_s": rbf_time}

    if fit["status"] != "completed":
        note = f"FD-inverse {fit.get('reason', 'failed')}; see e0/{data_name}/fd_inverse_runs.json"
        rows.append(evaluate.metrics_row(exp_id="E0", method="FD_inverse", data_file=data_file,
                                         status="failed", n_unknown=3,
                                         wall_time_s=fit["wall_time_s"], notes=note))
        write_metrics(rows)
        summary.update(status="failed", reason=note,
                       fd_inverse={k: v for k, v in fit.items() if k != "runs"})
        _write_summary(out_dir, summary)
        return summary

    xc, yc, th_fd, info_fd = fd.solve_field(fit["Bi"], known, nx=fit["nx"])
    theta_fd = fd.sample(xc, yc, th_fd, query).reshape(Xq.shape)
    flows_fd = fd.heat_flows(info_fd)
    bi_true, flows_true = evaluate.true_heat_flows(known, "E0 evaluation: true Bi and heat flows")
    q1_fd = evaluate.q1_metrics(theta_fd, ref, obs, known)
    bf_fd = evaluate.bi_and_flux_metrics(fit["Bi"], flows_fd, bi_true, flows_true)
    rows.append(evaluate.metrics_row(
        exp_id="E0", method="FD_inverse", data_file=data_file, status="completed",
        n_unknown=3, wall_time_s=fit["wall_time_s"], **bf_fd, **{k: q1_fd[k] for k in q1_cols},
        notes=f"101x81 grid, {fit['n_converged']}/{len(fit['runs'])} starts converged, "
              f"{fit['total_solves']} solves; field interpolated to 201x161 for Q1; flux from 101 grid"))
    write_metrics(rows)

    summary.update(status="completed",
                   fd_inverse={**{k: v for k, v in fit.items() if k != "runs"}, "q1": q1_fd,
                               "heat_flows": flows_fd, "metrics": bf_fd},
                   truth={"Bi": bi_true, "heat_flows_201": flows_true})
    _write_summary(out_dir, summary)
    plot_e0(ref, {"RBF (thin-plate)": theta_rbf, "FD-inverse": theta_fd}, obs, flows_true, flows_fd,
            known, fig_dir, data_name)
    return summary


# E1 is a tool diagnostic: true Bi fixed, no sensor data, independent of any inverse run.
E1_EXPERIMENT = {"seed": 0, "adam_steps": 5000, "lbfgs_iters": 100, "lambda_data": 0.0}
E1_TARGETS = {"rel_l2_max": 0.02, "energy_relerr_max": 0.01}


def plot_e1(ref, theta_hat, history, known, fig_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dT = common.dt_K(known)
    extent = [0, known["domain"]["LX"], 0, known["domain"]["LY"]]
    T_ref, T_hat = dT * ref["theta"], dT * theta_hat
    vmin, vmax = np.percentile(T_ref, [0.5, 99.5])
    E = T_hat - T_ref
    elim = np.abs(E).max()
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8), constrained_layout=True)
    for ax, (t, T) in zip(axes[:2], (("Reference (synthetic truth)", T_ref), ("PINN forward (true Bi fixed)", T_hat))):
        im = ax.imshow(T.T, origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap="inferno")
        ax.set_title(t)
    fig.colorbar(im, ax=axes[:2], label="T [°C]")
    im2 = axes[2].imshow(E.T, origin="lower", extent=extent, vmin=-elim, vmax=elim, cmap="RdBu_r")
    axes[2].set_title("PINN − reference")
    fig.colorbar(im2, ax=axes[2], label="ΔT [°C]")
    steps = [h["step"] for h in history]
    for key in ("loss", "pde", "bc"):
        axes[3].semilogy(steps, [h[key] for h in history], label=key)
    axes[3].set_xlabel("step"); axes[3].set_title("Training loss (raw MSE)"); axes[3].legend()
    fig.suptitle(title + " — simplified 2D model, synthetic data, diagnostic")
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)


def run_e1(seed=0):
    """Forward PINN with the true Bi fixed. Diagnostic only: truth is an input here."""
    import torch
    try:
        from . import pinn
    except ImportError:
        import src.pinn as pinn

    known = common.load_known()
    run = f"forward_seed{seed}"
    out_dir = common.RESULTS_DIR / "e1" / run
    fig_dir = common.RESULTS_DIR / "figures"
    for d in (out_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)
    exp = {**pinn.DEFAULT_EXPERIMENT, **E1_EXPERIMENT, "seed": seed}

    bi_true = evaluate.load_truth_bi("E1 forward diagnostic: true Bi fixed in the PINN (input)")
    torch.manual_seed(seed)
    model = pinn.HeatPINN(known, hidden=exp["hidden"], depth=exp["depth"], bi_fixed=bi_true)
    history, status = pinn.train(model, known, exp, obs=None)
    pinn.save_checkpoint(out_dir / "checkpoint.pt", model, exp, history, status,
                         extra={"exp_id": "E1", "bi_fixed_from": "config.json Bi_true (diagnostic)"})

    ref = evaluate.load_reference("E1 evaluation: forward field accuracy")
    theta_hat = pinn.predict_grid(model, ref["x"], ref["y"])
    dT = common.dt_K(known)
    err = theta_hat - ref["theta"]
    X, Y = np.meshgrid(ref["x"], ref["y"], indexing="ij")
    LX, LY = known["domain"]["LX"], known["domain"]["LY"]
    band = np.minimum.reduce([X, LX - X, Y, LY - Y]) <= evaluate.WALL_BAND
    field = {
        "rel_l2": float(np.linalg.norm(err) / np.linalg.norm(ref["theta"])),
        "rmse_C": float(dT * np.sqrt(np.mean(err ** 2))),
        "maxabs_C": float(dT * np.abs(err).max()),
        "wall_rmse_C": float(dT * np.sqrt(np.mean(err[band] ** 2))),
        "n_eval_points": int(err.size),
    }
    flows = pinn.heat_flows(model, known)
    _, flows_true = evaluate.true_heat_flows(known, "E1 evaluation: true heat flows")
    share_err = {w: 100 * (flows["share_signed"][w] - flows_true["share_signed"][w]) for w in common.ROBIN_WALLS}
    targets_met = {
        "rel_l2": field["rel_l2"] <= E1_TARGETS["rel_l2_max"],
        "energy": flows["energy_balance_relerr"] <= E1_TARGETS["energy_relerr_max"],
    }

    row = evaluate.metrics_row(
        exp_id="E1", method="PINN_forward", data_file="none", seed=seed, status=status["status"],
        n_unknown=0, **{f"flux_{w}": flows["flux"][w] for w in common.ROBIN_WALLS},
        source_integral=flows["source_integral"],
        **{f"share_{w}_err_pp": share_err[w] for w in common.ROBIN_WALLS},
        ranking_correct=flows["ranking"] == flows_true["ranking"],
        q1_rmse_C=field["rmse_C"], q1_maxabs_C=field["maxabs_C"], q1_wall_rmse_C=field["wall_rmse_C"],
        energy_balance_relerr=flows["energy_balance_relerr"], wall_time_s=status.get("wall_time_s"),
        notes=f"diagnostic: true Bi fixed, no sensor data; q1_* over the full 201x161 grid; "
              f"rel_L2={field['rel_l2']:.4g}; Adam {status.get('adam_steps_done')} + L-BFGS {exp['lbfgs_iters']}")
    write_metrics([row])

    summary = {
        "exp_id": "E1", "run": run, "status": status["status"], "environment": environment(),
        "torch": torch.__version__,
        "note": "Diagnostic forward solve with the TRUE Bi fixed (truth used as input). "
                "Not an inverse result; E2 starts from a fresh network with beta = 0.",
        "experiment": exp, "train_status": status, "targets": E1_TARGETS, "targets_met": targets_met,
        "field": field, "heat_flows": flows, "share_err_pp": share_err,
        "truth": {"Bi": bi_true, "heat_flows_201": flows_true},
        "n_parameters": pinn.count_parameters(model),
    }
    _write_summary(out_dir, summary)
    (out_dir / "history.json").write_text(json.dumps(common.to_jsonable(history), indent=1))
    plot_e1(ref, theta_hat, history, known, fig_dir / f"e1_{run}.png", f"E1 {run}")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", required=True, choices=["E0", "E1"])
    p.add_argument("--data", default="A_clean")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.exp == "E1":
        s = run_e1(args.seed)
        print(json.dumps(common.to_jsonable({
            "status": s["status"], "train_status": s["train_status"], "field": s["field"],
            "targets_met": s["targets_met"],
            "energy_balance_relerr": s["heat_flows"]["energy_balance_relerr"],
            "gradient_balance_relerr": s["heat_flows"]["gradient_balance_relerr"],
            "flux": s["heat_flows"]["flux"], "flux_grad": s["heat_flows"]["flux_grad"],
            "share_err_pp": s["share_err_pp"],
        }), indent=2))
        sys.exit(0 if s["status"] == "completed" else 2)
    if args.exp == "E0":
        s = run_e0(args.data)
        if s["status"] != "completed":
            print(json.dumps(common.to_jsonable({"status": s["status"], "reason": s["reason"],
                                                 "tool_checks_passed": s["tool_checks"]["passed"]}),
                             indent=2))
            sys.exit(2)
        print(json.dumps(common.to_jsonable({
            "tool_checks_passed": s["tool_checks"]["passed"],
            "fd_inverse_Bi": s["fd_inverse"]["Bi"], "truth_Bi": s["truth"]["Bi"],
            "fd_metrics": s["fd_inverse"]["metrics"], "fd_q1": s["fd_inverse"]["q1"],
            "rbf_q1": s["rbf"]["q1"], "fd_time_s": s["fd_inverse"]["wall_time_s"],
        }), indent=2))


if __name__ == "__main__":
    main()
