"""Experiment entry point.

    cd classroom_heat_pinn_data && python -m src.run_experiments --exp E0

Fitting (RBF, FD-inverse) receives only known.json and sensor observations.
Truth is loaded afterwards through evaluate.py and every access is logged.
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", required=True, choices=["E0"])
    p.add_argument("--data", default="A_clean")
    args = p.parse_args(argv)
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
