"""§18 stage 1: FD-inverse repeat baseline over conditions, sensor layouts and noise (LOG-117).

Fitting reads only synthetic/stage1/known_<cond>.json, sensors_<cond>_<layout>_clean.csv and
noise_standard_normal.npz. All fits finish before any truth file is opened; evaluation then reads
config_stage1.json and the reference fields. Results go to their own folder (not results/metrics.csv).

    python -B -m classroom_heat_pinn_data.src.stage1_fd --output-dir classroom_heat_pinn_data/results/stage1/pilot
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

try:
    from . import baselines, common, evaluate, extensions_fd as xf, fd
except ImportError:
    import src.baselines as baselines
    import src.common as common
    import src.evaluate as evaluate
    import src.extensions_fd as xf
    import src.fd as fd

STAGE1_DIR = common.PKG_DIR / "synthetic" / "stage1"
STAGE1_TRUTH_FILES = {"config_stage1.json", "reference_wall_eq.npz", "reference_wall_sink_eq.npz"}
CONDITIONS = ("base", "sink", "wall", "wall_sink", "wall_eq", "wall_sink_eq")
LAYOUTS = ("A", "C")
SIGMAS_C = (0.05, 0.1, 0.2)
N_PILOT = 10
# noise tables (training files): pilot (LOG-117) and independent confirmation (LOG-123)
NOISE_FILES = {"pilot": ("noise_standard_normal.npz", 10), "confirm": ("noise_confirm_standard_normal.npz", 30)}
MODEL_NX = 101
GRID_CHECK = (("wall", 201), ("wall_sink", 201), ("wall_eq", 201), ("wall_sink_eq", 201))
REL_BOUND_TOL = 1e-6


def load_known(cond, data_dir=STAGE1_DIR):
    known = json.loads((Path(data_dir) / f"known_{cond}.json").read_text())
    if any(k.lower().startswith("bi") for k in known):
        raise ValueError(f"known_{cond}.json unexpectedly contains Bi values")
    if isinstance(known.get("sink", {}).get("strength"), (int, float)):
        raise ValueError(f"known_{cond}.json must not contain the sink strength")
    return known


def load_observation(cond, layout, sigma_C=0.0, seed_index=None, data_dir=STAGE1_DIR, noise="pilot"):
    """Clean layout observations plus sigma * standard normal draw `seed_index` of the `noise` table."""
    if cond not in CONDITIONS or layout not in LAYOUTS:
        raise ValueError(f"unknown condition/layout {cond}/{layout}")
    known = load_known(cond, data_dir)
    name = f"sensors_{cond}_{layout}_clean.csv"
    df = pd.read_csv(Path(data_dir) / name)
    if list(df.columns) != common.SENSOR_COLUMNS:
        raise ValueError(f"{name}: unexpected columns {list(df.columns)}")
    xy, theta = df[["x", "y"]].to_numpy(float), df["theta_obs"].to_numpy(float)
    if not (np.isfinite(xy).all() and np.isfinite(theta).all()) or common.inside_solid(xy, known).any():
        raise ValueError(f"{name}: non-finite value or sensor inside a solid")
    tag = f"{cond}_{layout}_clean"
    if sigma_C:
        with np.load(Path(data_dir) / NOISE_FILES[noise][0]) as f:
            z = f["z"][seed_index]
        theta = theta + sigma_C / common.dt_K(known) * z
        tag = f"{cond}_{layout}_s{sigma_C:g}_{noise}{seed_index}"
    return known, common.Observations(tag, df["sensor_id"].to_numpy(), xy, theta, common.dt_K(known) * theta)


def tasks(mode="pilot"):
    """Every pre-registered fit: (cond, layout, sigma, seed_index, nx, noise table)."""
    if mode == "confirm":  # LOG-123: noisy only, fixed A/C, model grid 101
        n = NOISE_FILES["confirm"][1]
        return [(c, l, s, i, MODEL_NX, "confirm") for c in CONDITIONS for l in LAYOUTS for s in SIGMAS_C for i in range(n)]
    out = []
    for cond in CONDITIONS:
        for layout in LAYOUTS:
            out.append((cond, layout, 0.0, None, MODEL_NX, "pilot"))
            out += [(cond, layout, s, i, MODEL_NX, "pilot") for s in SIGMAS_C for i in range(N_PILOT)]
    out += [(cond, layout, 0.0, None, nx, "pilot") for cond, nx in GRID_CHECK for layout in LAYOUTS]
    return out


def fit_task(task):
    """One FD-inverse fit with the correct model; observations and known only."""
    cond, layout, sigma, seed, nx, noise = task
    known, obs = load_observation(cond, layout, sigma, seed, noise=noise)
    has_sink = common.sink_geometry(known) is not None
    t0 = time.perf_counter()
    fit = baselines.fit_fd_inverse(obs, known, {"sink": has_sink, "nx": nx})
    rec = {"cond": cond, "layout": layout, "sigma_C": sigma, "seed_index": seed, "noise": noise, "nx": nx, "has_sink": has_sink,
           "status": fit["status"], "fit_time_s": time.perf_counter() - t0, "n_converged": fit.get("n_converged")}
    if fit["status"] == "completed":
        beta = xf.params_of(fit, has_sink)
        names = [f"log Bi_{w}" for w in common.ROBIN_WALLS] + (["log f"] if has_sink else [])
        local = xf.local_std(xf.sensor_jacobian(beta, known, obs, has_sink, nx), common.dt_K(known), names)
        rec.update(Bi=fit["Bi"], f=fit["sink_fraction"], sensor_rms_C=fit["sensor_rms_C"],
                   local_rank=local["rank"], local_cond=local["condition_number"],
                   local_std=local.get("rel_std_0.1C"))
    return rec


def at_bound(value, bounds):
    return value is not None and any(abs(value / b - 1) < REL_BOUND_TOL for b in bounds)


def common_holdout_mask(refs):
    """Reference nodes that are fluid in every condition and are not a sensor of either layout
    (one mask for all conditions and layouts, so A and C are scored on the same points; LOG-121)."""
    first = next(iter(refs.values()))
    x, y = first["x"], first["y"]
    mask = np.ones(first["theta"].shape, bool)
    for r in refs.values():
        if "fluid" in r:
            mask &= r["fluid"].astype(bool)
    for layout in LAYOUTS:
        _, obs = load_observation("base", layout)
        nodes = fd.node_indices(x, y, obs.xy)
        mask[nodes] = False
    return mask


def evaluate_fits(records):
    """Truth is read here, after every fit has finished. Returns the SHA-256 of every evaluation file."""
    cfg_path = STAGE1_DIR / "config_stage1.json"
    evaluate._log_access("config_stage1.json", "Stage 1 evaluation: true Bi, f and reference pointers")
    cfg = json.loads(cfg_path.read_text())
    used = {cfg_path: None}
    refs = {}
    for cond in CONDITIONS:
        path = common.PKG_DIR / cfg["conditions"][cond]["reference"]  # relative to classroom_heat_pinn_data/
        evaluate._log_access(path.name, f"Stage 1 evaluation: Q1 reference ({cond})")
        with np.load(path) as f:
            refs[cond] = {k: f[k] for k in f.files}
        used[path] = None
    root = common.PKG_DIR.parent
    eval_sha = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in used}
    holdout = common_holdout_mask(refs)
    rows = []
    for rec in records:
        row = dict(rec)
        if rec["status"] == "completed":
            truth = cfg["conditions"][rec["cond"]]
            known, obs = load_observation(rec["cond"], rec["layout"], rec["sigma_C"], rec["seed_index"], noise=rec["noise"])
            _, _, th, _ = fd.solve_field(rec["Bi"], known, nx=xf.EVAL_NX, sink_fraction=rec["f"])
            q1 = evaluate.q1_metrics(th, refs[rec["cond"]], obs, known)
            err = (th - refs[rec["cond"]]["theta"]) * common.dt_K(known)
            row["q1_common_rmse_C"] = float(np.sqrt(np.mean(err[holdout] ** 2)))
            scale = rec["sigma_C"] / 0.1  # local std is stored at 0.1 C; it scales linearly with sigma
            for w in common.ROBIN_WALLS:
                row[f"Bi_{w}_relerr"] = rec["Bi"][w] / truth["Bi"][w] - 1
                row[f"Bi_{w}_logerr"] = float(np.log(rec["Bi"][w] / truth["Bi"][w]))
                row[f"Bi_{w}_local_std"] = rec["local_std"][f"log Bi_{w}"] * scale if rec["local_std"] else None
            if rec["has_sink"]:
                row["f_relerr"] = rec["f"] / truth["f"] - 1
                row["f_logerr"] = float(np.log(rec["f"] / truth["f"]))
                row["f_local_std"] = rec["local_std"]["log f"] * scale if rec["local_std"] else None
            row["at_bound"] = (any(at_bound(rec["Bi"][w], baselines.BI_BOUNDS) for w in common.ROBIN_WALLS)
                               or (rec["has_sink"] and at_bound(rec["f"], baselines.SINK_BOUNDS)))
            row.update({k: q1.get(k) for k in ("q1_rmse_C", "q1_maxabs_C", "q1_rmse_wall_tip_C", "q1_rmse_outside_wall_tip_C")})
        rows.append(row)
    return cfg, rows, eval_sha, int(holdout.sum())


def aggregate(rows):
    """Per (cond, layout, sigma, nx): bias, RMSE, median, p90 of relative errors, coverage, failures."""
    groups = {}
    for r in rows:
        groups.setdefault((r["cond"], r["layout"], r["sigma_C"], r["nx"]), []).append(r)
    out = []
    for (cond, layout, sigma, nx), rs in sorted(groups.items()):
        ok = [r for r in rs if r["status"] == "completed"]
        g = {"cond": cond, "layout": layout, "sigma_C": sigma, "nx": nx, "n": len(rs), "n_failed": len(rs) - len(ok),
             "at_bound_rate": float(np.mean([r["at_bound"] for r in ok])) if ok else None,
             "fit_time_s_mean": float(np.mean([r["fit_time_s"] for r in rs]))}
        params = [f"Bi_{w}" for w in common.ROBIN_WALLS] + (["f"] if ok and ok[0]["has_sink"] else [])
        for p in params:
            rel = np.array([r[f"{p}_relerr"] for r in ok])
            g[f"{p}_bias"] = float(rel.mean()) if len(rel) else None
            g[f"{p}_rmse"] = float(np.sqrt(np.mean(rel ** 2))) if len(rel) else None
            g[f"{p}_median"] = float(np.median(rel)) if len(rel) else None
            g[f"{p}_p90_abs"] = float(np.quantile(np.abs(rel), 0.9)) if len(rel) else None
            pairs = [(r[f"{p}_logerr"], r[f"{p}_local_std"]) for r in ok if r.get(f"{p}_local_std")]
            if sigma and pairs:
                g[f"{p}_coverage_1.96"] = float(np.mean([abs(e) <= 1.96 * s for e, s in pairs]))
                g[f"{p}_local_std_mean"] = float(np.mean([s for _, s in pairs]))
        qc = [r["q1_common_rmse_C"] for r in ok if r.get("q1_common_rmse_C") is not None]
        g["q1_common_rmse_C_mean"] = float(np.mean(qc)) if qc else None
        q1 = np.array([r["q1_rmse_C"] for r in ok]) if ok else np.array([])
        g["q1_rmse_C_mean"] = float(q1.mean()) if len(q1) else None
        g["q1_rmse_C_p90"] = float(np.quantile(q1, 0.9)) if len(q1) else None
        tips = [r["q1_rmse_wall_tip_C"] for r in ok if r.get("q1_rmse_wall_tip_C") is not None]
        g["q1_wall_tip_mean"] = float(np.mean(tips)) if tips else None
        out.append(g)
    return out


def provenance(files):
    root = common.PKG_DIR.parent
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
        except Exception as exc:  # recorded, not fatal
            return f"unavailable: {exc}"
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "worktree_diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
            "input_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
            "bounds": {"Bi": baselines.BI_BOUNDS, "f": baselines.SINK_BOUNDS},
            "starts": {"no_sink": baselines.DEFAULT_STARTS, "sink": baselines.SINK_STARTS},
            "model_nx": MODEL_NX, "eval_nx": xf.EVAL_NX, "sigmas_C": SIGMAS_C, "n_pilot": N_PILOT}


def run(output_dir, workers=4, mode="pilot"):
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already holds results; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in STAGE1_DIR.iterdir() if p.name not in STAGE1_TRUTH_FILES and p.name != "verification_stage1.json")
    prov = provenance(inputs + [Path(__file__), Path(baselines.__file__), Path(fd.__file__), Path(xf.__file__)])
    t0 = time.perf_counter()
    todo = tasks(mode)
    with ProcessPoolExecutor(max_workers=workers) as pool:  # fitting: no truth access
        records = list(pool.map(fit_task, todo, chunksize=4))
    fit_wall = time.perf_counter() - t0
    cfg, rows, eval_sha, n_holdout = evaluate_fits(records)
    groups = aggregate(rows)
    summary = {"preregistration": "LOG-117" if mode == "pilot" else "LOG-123", "mode": mode,
               "evaluation_sha256": eval_sha, "common_holdout_nodes": n_holdout, "n_fits": len(rows), "n_failed": sum(r["status"] != "completed" for r in rows),
               "fit_wall_time_s": fit_wall, "workers": workers, "provenance": prov, "groups": groups}
    (out / "summary.json").write_text(json.dumps(common.to_jsonable(summary), indent=1))
    cols = sorted({k for r in rows for k in r if k not in ("Bi", "local_std")})
    with open(out / "fits.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    with open(out / "groups.csv", "w", newline="") as fh:
        gcols = sorted({k for g in groups for k in g})
        w = csv.DictWriter(fh, gcols)
        w.writeheader()
        for g in groups:
            w.writerow(g)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--mode", choices=("pilot", "confirm"), default="pilot")
    args = p.parse_args(argv)
    try:
        s = run(args.output_dir, args.workers, args.mode)
    except FileExistsError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        sys.exit(3)
    print(json.dumps({"n_fits": s["n_fits"], "n_failed": s["n_failed"], "fit_wall_time_s": s["fit_wall_time_s"]}))
    sys.exit(0 if s["n_failed"] == 0 else 2)


if __name__ == "__main__":
    main()
