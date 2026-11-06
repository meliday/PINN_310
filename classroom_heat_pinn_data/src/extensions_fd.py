"""FD-only runs for the user-requested extensions (LOG-088): X1-d and the X2 four-condition comparison.

X1-d: FD-inverse of Bi on the internal-wall data (wall geometry known), model grid 201 and 101,
plus a wall-ignored control. X2: base / sink only / wall only / wall + sink, clean and noisy,
model grid 101 for every condition, each fitted with the correct model and with controls that
ignore the wall and/or the sink. Fitting reads only known_*.json and sensors_*.csv; truth is
read afterwards and every read is logged in results/logs/truth_access.log.

Evaluation field: the FD solution at the estimated parameters re-solved on the 201 grid of the
saved reference (the reference grid), so no interpolation across the wall is needed.
"""
import json
import time

import numpy as np

try:
    from . import baselines, common, evaluate, fd
except ImportError:
    import src.baselines as baselines
    import src.common as common
    import src.evaluate as evaluate
    import src.fd as fd

EVAL_NX = 201
X2_NX = 101
SIGMA_C = 0.1

# condition -> loader of known, data names, truth files, which physics the truth contains
CONDITIONS = {
    "base": {"known": common.load_known, "data": ("A_clean", "A_noisy"), "wall": False, "sink": False},
    "sink": {"known": common.load_known_e5b, "data": ("A_sink_clean", "A_sink_noisy"), "wall": False, "sink": True,
             "dir": common.E5B_DIR, "config": "config_e5b.json", "reference": "reference_e5b.npz"},
    "wall": {"known": common.load_known_x1, "data": ("A_x1_clean", "A_x1_noisy"), "wall": True, "sink": False,
             "dir": common.X1_DIR, "config": "config_x1.json", "reference": "reference_x1.npz"},
    "wall_sink": {"known": common.load_known_x2, "data": ("A_x2_clean", "A_x2_noisy"), "wall": True, "sink": True,
                  "dir": common.X2_DIR, "config": "config_x2.json", "reference": "reference_x2.npz"},
}


def model_variants(cond):
    """Model variants fitted for a condition: (name, use_wall, use_sink); the correct model first."""
    c = CONDITIONS[cond]
    out = [("correct", c["wall"], c["sink"])]
    if c["wall"] or c["sink"]:
        out.append(("ignore_all", False, False))
    if c["wall"] and c["sink"]:
        out += [("ignore_wall", False, True), ("ignore_sink", True, False)]
    return out


def variant_known(known, use_wall, use_sink):
    k = {key: v for key, v in known.items() if key not in ("solids", "sink")}
    if use_wall:
        k["solids"] = known["solids"]
    if use_sink:
        k["sink"] = known["sink"]
    return k


def params_of(fit, use_sink):
    p = [fit["Bi"][w] for w in common.ROBIN_WALLS]
    return np.log(p + ([fit["sink_fraction"]] if use_sink else []))


def sensor_jacobian(beta, known, obs, use_sink, nx, h=1e-4):
    """d theta_sensor / d (log Bi, [log f]) by central differences at beta."""
    def pred(b):
        f = float(np.exp(b[3])) if use_sink else None
        x, y, th, _ = fd.solve_field(np.exp(b[:3]), known, nx=nx, sink_fraction=f)
        return fd.sample(x, y, th, obs.xy)
    J = np.empty((len(obs.theta), len(beta)))
    for i in range(len(beta)):
        e = np.zeros(len(beta)); e[i] = h
        J[:, i] = (pred(beta + e) - pred(beta - e)) / (2 * h)
    return J


def local_std(J, dT, names):
    s = np.linalg.svd(J, compute_uv=False)
    out = {"rank": int(np.sum(s > s[0] * 1e-10)), "condition_number": float(s[0] / s[-1])}
    if out["rank"] == len(names):
        cov = np.linalg.inv(J.T @ J) * (SIGMA_C / dT) ** 2
        d = np.sqrt(np.diag(cov))
        out["rel_std_0.1C"] = dict(zip(names, d))
        out["correlation"] = (cov / np.outer(d, d)).tolist()
    return out


def load_truth(cond, purpose):
    """True Bi, sink fraction, signed wall shares of the heater (fluid) power, and the 201 reference."""
    c = CONDITIONS[cond]
    if cond == "base":
        bi_true, flows = evaluate.true_heat_flows(common.load_known(), f"{purpose}: true Bi and heat flows")
        ref = evaluate.load_reference(f"{purpose}: Q1 reference")
        return {"Bi": bi_true, "f": None, "share": flows["share_signed"]}, ref
    evaluate._log_access(c["config"], f"{purpose}: true Bi, sink fraction, wall shares")
    cfg = json.loads((c["dir"] / c["config"]).read_text())
    evaluate._log_access(c["reference"], f"{purpose}: Q1 reference")
    with np.load(c["dir"] / c["reference"]) as f:
        ref = {k: f[k] for k in f.files}
    f_true = cfg["sink"]["fraction_of_heater"] if c["sink"] else None
    return {"Bi": common.bi_dict(cfg["Bi_true"]), "f": f_true, "share": cfg["wall_flow_share_of_heater_true"]}, ref


def fit_condition(cond, nx, variants=None):
    """Fit every (data, variant) of a condition. Observations + known only; no truth here."""
    c = CONDITIONS[cond]
    known = c["known"]()
    fits = []
    for name in c["data"]:
        obs = common.load_observations(name, known)
        for vname, use_wall, use_sink in (variants or model_variants(cond)):
            k = variant_known(known, use_wall, use_sink)
            t0 = time.perf_counter()
            fit = baselines.fit_fd_inverse(obs, k, {"sink": use_sink, "nx": nx})
            rec = {"condition": cond, "data": name, "variant": vname, "use_wall": use_wall, "use_sink": use_sink,
                   "nx": nx, "fit": fit, "fit_time_s": time.perf_counter() - t0}
            if fit["status"] == "completed":
                beta = params_of(fit, use_sink)
                names = [f"log Bi_{w}" for w in common.ROBIN_WALLS] + (["log f"] if use_sink else [])
                rec["local"] = local_std(sensor_jacobian(beta, k, obs, use_sink, nx), common.dt_K(known), names)
            fits.append(rec)
    return known, fits


def evaluate_fits(cond, known, fits, purpose):
    """Truth is read here, after all fits of the condition."""
    truth, ref = load_truth(cond, purpose)
    rows = []
    for rec in fits:
        fit = rec["fit"]
        out = {k: rec[k] for k in ("condition", "data", "variant", "use_wall", "use_sink", "nx", "fit_time_s")}
        out["status"] = fit["status"]
        out["n_converged"] = fit.get("n_converged")
        if fit["status"] != "completed":
            rows.append(out)
            continue
        k = variant_known(known, rec["use_wall"], rec["use_sink"])
        obs = common.load_observations(rec["data"], known)
        x, y, th, info = fd.solve_field(fit["Bi"], k, nx=EVAL_NX, sink_fraction=fit["sink_fraction"])
        flows = fd.heat_flows(info)
        out.update({
            "Bi": fit["Bi"], "Bi_relerr": {w: fit["Bi"][w] / truth["Bi"][w] - 1 for w in common.ROBIN_WALLS},
            "f": fit["sink_fraction"],
            "f_relerr": (fit["sink_fraction"] / truth["f"] - 1) if (truth["f"] and fit["sink_fraction"]) else None,
            "sensor_rms_C": fit["sensor_rms_C"], "q1": evaluate.q1_metrics(th, ref, obs, known),
            "share": flows["share_signed"], "share_sink": flows.get("share_sink"),
            "share_err_pp": {w: 100 * (flows["share_signed"][w] - truth["share"][w]) for w in common.ROBIN_WALLS},
            "energy_balance_relerr": flows["energy_balance_relerr"], "local": rec.get("local"),
        })
        rows.append(out)
    return truth, rows


def metrics_rows(exp_id, rows):
    out = []
    for r in rows:
        method = f"FD_{r['variant']}_nx{r['nx']}"
        data_file = f"sensors_{r['data']}.csv"
        if r["status"] != "completed":
            out.append(evaluate.metrics_row(exp_id=exp_id, method=method, data_file=data_file, status="failed",
                                            n_unknown=4 if r["use_sink"] else 3, wall_time_s=r["fit_time_s"],
                                            notes=f"{r['condition']}: FD-inverse failed"))
            continue
        out.append(evaluate.metrics_row(
            exp_id=exp_id, method=method, data_file=data_file, status="completed",
            n_unknown=4 if r["use_sink"] else 3,
            **{f"bi_{w}_hat": r["Bi"][w] for w in common.ROBIN_WALLS},
            **{f"bi_{w}_relerr": r["Bi_relerr"][w] for w in common.ROBIN_WALLS},
            **{f"share_{w}_err_pp": r["share_err_pp"][w] for w in common.ROBIN_WALLS},
            q1_rmse_C=r["q1"]["q1_rmse_C"], q1_maxabs_C=r["q1"]["q1_maxabs_C"], q1_wall_rmse_C=r["q1"]["q1_wall_rmse_C"],
            energy_balance_relerr=r["energy_balance_relerr"], wall_time_s=r["fit_time_s"],
            notes=(f"{r['condition']}: wall={'yes' if r['use_wall'] else 'no'} sink={'f' if r['use_sink'] else 'no'}"
                   + (f"; f_hat={r['f']:.6g} (relerr {r['f_relerr']:+.4f})" if r["f"] else "")
                   + f"; sensor RMS {r['sensor_rms_C']:.4g} C; {r['n_converged']} starts converged")))
    return out


def plot_conditions(results, fig_path, title):
    """Truth field (clean) and correct-model FD error for each condition, same colour limits per row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    conds = list(results)
    fig, axes = plt.subplots(2, len(conds), figsize=(4.6 * len(conds), 7), constrained_layout=True, squeeze=False)
    T_all = [results[c]["T_ref"] for c in conds]
    vmin, vmax = min(np.nanmin(t) for t in T_all), max(np.nanmax(t) for t in T_all)
    elim = max(np.nanmax(np.abs(results[c]["err"])) for c in conds)
    for j, c in enumerate(conds):
        r = results[c]
        for i, (A, cmap, lo, hi, lab) in enumerate(((r["T_ref"], "inferno", vmin, vmax, "truth °C"),
                                                    (r["err"], "RdBu_r", -elim, elim, "FD − truth °C"))):
            ax = axes[i, j]
            im = ax.imshow(A.T, origin="lower", extent=[0, 1, 0, 0.8], cmap=cmap, vmin=lo, vmax=hi)
            for s in r["solids"]:
                ax.add_patch(Rectangle((s["x0"], s["y0"]), s["x1"] - s["x0"], s["y1"] - s["y0"], color="0.6"))
            if r["sink"]:
                ax.plot(r["sink"]["x"], r["sink"]["y"], "c*", ms=10)
            ax.plot(r["sensors"][:, 0], r["sensors"][:, 1], "w.", ms=5)
            ax.set_title(f"{c}: {lab}")
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(title)
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)


def _field_for_plot(cond, known, row, ref):
    k = variant_known(known, row["use_wall"], row["use_sink"])
    _, _, th, _ = fd.solve_field(row["Bi"], k, nx=EVAL_NX, sink_fraction=row["f"])
    dT = common.dt_K(known)
    T_ref = dT * np.where(ref.get("fluid", np.ones_like(ref["theta"], bool)), ref["theta"], np.nan)
    return {"T_ref": T_ref, "err": dT * th - T_ref, "solids": common.solids(known),
            "sink": common.sink_geometry(known), "sensors": common.load_observations(row["data"], known).xy}


def run_x1_fd(write_metrics, claim_run_dir, environment, overwrite=False):
    """X1-d: Bi with the wall known, nx 201 and 101, and a wall-ignored control (nx 101)."""
    out_dir = common.RESULTS_DIR / "x1" / "fd_A_x1"
    claim_run_dir(out_dir, overwrite)
    known, fits = fit_condition("wall", 201, [("correct", True, False)])
    fits += fit_condition("wall", 101, [("correct", True, False), ("ignore_wall", False, False)])[1]
    truth, rows = evaluate_fits("wall", known, fits, "X1-d evaluation")
    write_metrics(metrics_rows("X1", rows))
    targets = {"clean_nx201_bi_relerr_max": 0.02, "clean_nx201_q1_rmse_C_max": 0.1}
    clean201 = next(r for r in rows if r["data"] == "A_x1_clean" and r["nx"] == 201)
    met = (clean201["status"] == "completed"
           and max(abs(v) for v in clean201["Bi_relerr"].values()) <= targets["clean_nx201_bi_relerr_max"]
           and clean201["q1"]["q1_rmse_C"] <= targets["clean_nx201_q1_rmse_C_max"])
    summary = {"exp_id": "X1", "run": "fd_A_x1", "preregistration": "LOG-088", "environment": environment(),
               "status": "completed" if all(r["status"] == "completed" for r in rows) else "failed",
               "targets": targets, "targets_met": met, "truth": truth, "results": rows}
    (out_dir / "summary.json").write_text(json.dumps(common.to_jsonable(summary), indent=1))
    return summary


def run_x2_fd(write_metrics, claim_run_dir, environment, overwrite=False):
    """X2: four conditions x clean/noisy, model grid 101, correct model + controls."""
    out_dir = common.RESULTS_DIR / "x2" / "fd_four_conditions"
    claim_run_dir(out_dir, overwrite)
    all_rows, truths, plots = [], {}, {}
    for cond in CONDITIONS:  # fit then evaluate per condition; no truth is read while fitting
        known, fits = fit_condition(cond, X2_NX)
        truth, rows = evaluate_fits(cond, known, fits, f"X2 evaluation ({cond})")
        truths[cond] = truth
        all_rows += rows
        clean = next(r for r in rows if r["variant"] == "correct" and r["data"].endswith("clean"))
        if clean["status"] == "completed":
            _, ref = load_truth(cond, f"X2 figure ({cond})")
            plots[cond] = _field_for_plot(cond, known, clean, ref)
    write_metrics(metrics_rows("X2", all_rows))
    summary = {"exp_id": "X2", "run": "fd_four_conditions", "preregistration": "LOG-088",
               "environment": environment(), "model_nx": X2_NX, "eval_nx": EVAL_NX,
               "status": "completed" if all(r["status"] == "completed" for r in all_rows) else "failed",
               "heat_basis": "heater amplitude fixed; shares are fractions of each condition's fluid heater power",
               "truth": truths, "results": all_rows}
    (out_dir / "summary.json").write_text(json.dumps(common.to_jsonable(summary), indent=1))
    _write_table(out_dir / "table.csv", all_rows)
    if plots:
        plot_conditions(plots, common.RESULTS_DIR / "figures" / "x2_fd_four_conditions.png",
                        "X2: FD-inverse (correct model, clean), simplified 2D model, synthetic, single realisation")
    return summary


def _write_table(path, rows):
    import csv
    cols = ["condition", "data", "variant", "nx", "status", "n_converged", "Bi_S_relerr", "Bi_E_relerr", "Bi_N_relerr",
            "f", "f_relerr", "sensor_rms_C", "q1_rmse_C", "q1_maxabs_C", "q1_rmse_wall_tip_C",
            "share_S_err_pp", "share_E_err_pp", "share_N_err_pp", "share_sink", "energy_balance_relerr",
            "condition_number", "fit_time_s"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for r in rows:
            d = {k: r.get(k) for k in ("condition", "data", "variant", "nx", "status", "n_converged", "f", "f_relerr",
                                       "sensor_rms_C", "share_sink", "energy_balance_relerr", "fit_time_s")}
            if r["status"] == "completed":
                d.update({f"Bi_{k}_relerr": r["Bi_relerr"][k] for k in common.ROBIN_WALLS})
                d.update({f"share_{k}_err_pp": r["share_err_pp"][k] for k in common.ROBIN_WALLS})
                d.update({k: r["q1"].get(k) for k in ("q1_rmse_C", "q1_maxabs_C", "q1_rmse_wall_tip_C")})
                d["condition_number"] = (r.get("local") or {}).get("condition_number")
            w.writerow(d)
