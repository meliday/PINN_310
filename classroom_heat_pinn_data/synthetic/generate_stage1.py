"""Generate the §18 stage-1 FD repeat-baseline inputs (pre-registered in LOG-117).

Six data conditions: base, sink, wall, wall_sink (heater amplitude fixed, existing truth) and
wall_eq, wall_sink_eq (the heater amplitude scaled so the fluid region receives the same heater
power as the open room; truth solved on nx = 1601 and stored at the 201-grid nodes). Two sensor
layouts: A (existing) and C (new, greedy D-optimal over the four amplitude-fixed conditions at the
generating coefficients -- a development choice). Clean sensor values are read from the 201-node
truth fields; noise is a table of standard normal draws (seeds 31000 + i) scaled by sigma in the runner.

Outputs (synthetic/stage1 by default; an existing directory with files is refused):
  known_<cond>.json                 what estimation may use (Bi and sink strength unknown)
  sensors_<cond>_<layout>_clean.csv clean observations (training)
  noise_standard_normal.npz         z[seed, sensor] (training)
  layouts.json                      sensor coordinates and the layout-C design record (training)
  config_stage1.json                truth: Bi, f, heater scaling, file pointers (evaluation only)
  reference_wall_eq.npz, reference_wall_sink_eq.npz   truth fields (evaluation only)
  verification_stage1.json          integrity checks
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

if __package__:
    from .generate_data import BI_TRUE, DT, HEATER, LAYOUT_A, THETA_INF, write_sensors
    from .generate_e5b import SINK, SINK_FRACTION
    from .generate_x1 import NX_TRUTH, WALL, tip_mask
    from .solver import heater_source, make_grid
    from . import solver_wall
else:
    from generate_data import BI_TRUE, DT, HEATER, LAYOUT_A, THETA_INF, write_sensors
    from generate_e5b import SINK, SINK_FRACTION
    from generate_x1 import NX_TRUTH, WALL, tip_mask
    from solver import heater_source, make_grid
    import solver_wall

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "stage1"
NX_REF, NX_MODEL = 201, 101
NOISE_SEEDS = tuple(31000 + i for i in range(10))
CONFIRM_SEEDS = tuple(32000 + i for i in range(30))  # LOG-123: independent confirmation draws
AMPLITUDE_CONDITIONS = {  # cond -> (folder, known file, reference file, has wall, has sink)
    "base": ("data", "known.json", "reference.npz", False, False),
    "sink": ("e5b", "known_e5b.json", "reference_e5b.npz", False, True),
    "wall": ("x1", "known_x1.json", "reference_x1.npz", True, False),
    "wall_sink": ("x2", "known_x2.json", "reference_x2.npz", True, True),
}
EQ_CONDITIONS = {"wall_eq": "wall", "wall_sink_eq": "wall_sink"}
DESIGN = {"spacing": 0.05, "x_range": (0.05, 0.95), "y_range": (0.05, 0.75), "min_distance": 0.1,
          "wall_clearance": 0.02, "n_sensors": 10, "ridge": 1e-10, "step": 1e-4}


def fluid_power(nx, A_scale, sink_fraction=None):
    """Heater (and sink) source on an nx grid for the wall room, heater amplitude scaled."""
    x, y, _ = make_grid(nx)
    X, Y = np.meshgrid(x, y, indexing="ij")
    vol = solver_wall.fluid_volumes(x, y, [WALL])
    q_heat = heater_source(X, Y, [{**HEATER, "A": HEATER["A"] * A_scale}])
    H = float((q_heat * vol).sum())
    Q = q_heat
    if sink_fraction is not None:
        shape = np.exp(-((X - SINK["x"]) ** 2 + (Y - SINK["y"]) ** 2) / (2 * SINK["s"] ** 2))
        Q = q_heat - sink_fraction * H / float((shape * vol).sum()) * shape
    return Q, H


def open_room_power(nx):
    x, y, h = make_grid(nx)
    X, Y = np.meshgrid(x, y, indexing="ij")
    wx = np.full(len(x), h); wx[[0, -1]] = h / 2
    wy = np.full(len(y), h); wy[[0, -1]] = h / 2
    return float((heater_source(X, Y, [HEATER]) * np.outer(wx, wy)).sum())


def candidates():
    d = DESIGN
    xs = np.round(np.arange(d["x_range"][0], d["x_range"][1] + 1e-9, d["spacing"]), 2)
    ys = np.round(np.arange(d["y_range"][0], d["y_range"][1] + 1e-9, d["spacing"]), 2)
    pts = np.array([(x, y) for x in xs for y in ys])
    c = d["wall_clearance"]
    near_wall = (pts[:, 0] <= WALL["x1"] + c) & (pts[:, 1] >= WALL["y0"] - c) & (pts[:, 1] <= WALL["y1"] + c)
    return pts[~near_wall]


def field_jacobian(known, has_sink, nx=NX_MODEL):
    """d theta / d (log Bi, [log f]) on the nx grid at the generating coefficients (central differences)."""
    import sys
    sys.path.insert(0, str(HERE.parents[1]))
    from classroom_heat_pinn_data.src import fd
    beta = np.log([BI_TRUE[w] for w in "SEN"] + ([SINK_FRACTION] if has_sink else []))
    def solve(b):
        _, _, th, _ = fd.solve_field(np.exp(b[:3]), known, nx=nx, sink_fraction=float(np.exp(b[3])) if has_sink else None)
        return th
    h = DESIGN["step"]
    cols = []
    for i in range(len(beta)):
        e = np.zeros(len(beta)); e[i] = h
        cols.append((solve(beta + e) - solve(beta - e)) / (2 * h))
    x, y, _ = make_grid(nx)
    return x, y, np.stack(cols, axis=-1)


def design_layout_c(knowns):
    """Greedy D-optimal 10-sensor layout over the four amplitude-fixed conditions."""
    cand = candidates()
    jac = {}
    for cond, (_, _, _, wall, sink) in AMPLITUDE_CONDITIONS.items():
        x, y, J = field_jacobian(knowns[cond], sink)
        ix = np.rint(cand[:, 0] / (x[1] - x[0])).astype(int)
        iy = np.rint(cand[:, 1] / (y[1] - y[0])).astype(int)
        jac[cond] = J[ix, iy]  # (n_candidates, n_params)
        assert np.isfinite(jac[cond]).all()
    chosen = []
    history = []
    for _ in range(DESIGN["n_sensors"]):
        best, best_score = None, -np.inf
        for k in range(len(cand)):
            if k in chosen or any(np.hypot(*(cand[k] - cand[j])) < DESIGN["min_distance"] - 1e-9 for j in chosen):
                continue
            rows = chosen + [k]
            score = 0.0
            for J in jac.values():
                M = J[rows].T @ J[rows] + DESIGN["ridge"] * np.eye(J.shape[1])
                score += np.linalg.slogdet(M)[1]
            if score > best_score:
                best, best_score = k, score
        chosen.append(best)
        history.append({"point": cand[best].tolist(), "criterion": best_score})
    layout = cand[chosen]
    final = {cond: float(np.linalg.slogdet(J[chosen].T @ J[chosen])[1]) for cond, J in jac.items()}
    ix_a = [int(np.argmin(np.hypot(cand[:, 0] - px, cand[:, 1] - py))) for px, py in LAYOUT_A]
    return layout, {"n_candidates": len(cand), "greedy": history, "logdet_C": final, "rule": DESIGN}


def sensor_values(ref, pts):
    x, y = ref["x"], ref["y"]
    ix = np.rint(pts[:, 0] / (x[1] - x[0])).astype(int)
    iy = np.rint(pts[:, 1] / (y[1] - y[0])).astype(int)
    assert np.allclose(x[ix], pts[:, 0]) and np.allclose(y[iy], pts[:, 1]), "sensors must sit on reference nodes"
    vals = ref["theta"][ix, iy]
    assert np.isfinite(vals).all(), "sensor inside a solid"
    return vals


def generate(output_dir=DEFAULT_OUTPUT):
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already holds files; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)

    knowns, refs, truth = {}, {}, {}
    for cond, (folder, kname, rname, wall, sink) in AMPLITUDE_CONDITIONS.items():
        knowns[cond] = json.loads((HERE / folder / kname).read_text())
        with np.load(HERE / folder / rname) as f:
            refs[cond] = {k: f[k] for k in f.files}
        truth[cond] = {"Bi": BI_TRUE, "f": SINK_FRACTION if sink else None, "reference": f"synthetic/{folder}/{rname}",
                       "heater_A": HEATER["A"], "basis": "amplitude"}

    # equal effective heater power: scale A so the fluid region gets the open-room heater power (1601 FV)
    H_open = open_room_power(NX_TRUTH)
    _, H_wall = fluid_power(NX_TRUTH, 1.0)
    scale = H_open / H_wall
    for cond, base_cond in EQ_CONDITIONS.items():
        sink = AMPLITUDE_CONDITIONS[base_cond][4]
        Q, H = fluid_power(NX_TRUTH, scale, SINK_FRACTION if sink else None)
        x, y, theta, info = solver_wall.solve(NX_TRUTH, BI_TRUE, THETA_INF, solids=[WALL], Q=Q)
        s = (NX_TRUTH - 1) // (NX_REF - 1)
        xr, yr = x[::s], y[::s]
        ref = {"x": xr, "y": yr, "theta": theta[::s, ::s], "fluid": info["active"][::s, ::s], "wall_tip": tip_mask(xr, yr)}
        np.savez(out / f"reference_{cond}.npz", **ref, T_C=DT * ref["theta"])
        refs[cond] = ref
        k = copy.deepcopy(knowns[base_cond])
        k["heater"] = {**k["heater"], "A": HEATER["A"] * scale}
        k["_note"] = (k.get("_note", "") + " Stage 1: heater amplitude scaled so the fluid region receives the "
                      "open-room heater power (equal effective heat input).").strip()
        knowns[cond] = k
        truth[cond] = {"Bi": BI_TRUE, "f": SINK_FRACTION if sink else None, "reference": f"synthetic/stage1/reference_{cond}.npz",
                       "heater_A": HEATER["A"] * scale, "basis": "equal_heat", "heater_power_fluid_1601": H,
                       "energy_balance_rel_err_1601": info["balance_rel_err"],
                       "sink_fraction_realised_1601": (H - info["q_in"]) / H if sink else None}

    layout_c, design = design_layout_c(knowns)
    layouts = {"A": np.asarray(LAYOUT_A, float), "C": layout_c}
    for cond in list(AMPLITUDE_CONDITIONS) + list(EQ_CONDITIONS):
        (out / f"known_{cond}.json").write_text(json.dumps(knowns[cond], indent=2))
        for lname, pts in layouts.items():
            write_sensors(out / f"sensors_{cond}_{lname}_clean.csv", pts, sensor_values(refs[cond], pts))
    z = np.stack([np.random.default_rng(seed).standard_normal(10) for seed in NOISE_SEEDS])
    np.savez(out / "noise_standard_normal.npz", z=z, seeds=np.array(NOISE_SEEDS))
    (out / "layouts.json").write_text(json.dumps({k: v.tolist() for k, v in layouts.items()} | {"design_C": design}, indent=1))
    config = {"_note": "Stage-1 truth. Generation / evaluation only; estimation code must not read this file.",
              "preregistration": "LOG-117", "conditions": truth, "equal_heat_scale": scale,
              "heater_power_open_1601": H_open, "heater_power_wall_amplitude_1601": H_wall,
              "noise_seeds": NOISE_SEEDS, "sigmas_C": [0.05, 0.1, 0.2], "DT_K": DT}
    (out / "config_stage1.json").write_text(json.dumps(config, indent=2))

    # ---- verification
    existing = {"base": HERE / "data/sensors_A_clean.csv", "sink": HERE / "e5b/sensors_A_sink_clean.csv",
                "wall": HERE / "x1/sensors_A_x1_clean.csv", "wall_sink": HERE / "x2/sensors_A_x2_clean.csv"}
    same_a = {c: (out / f"sensors_{c}_A_clean.csv").read_bytes() == p.read_bytes() for c, p in existing.items()}
    ver = {
        "layout_A_clean_equals_existing": same_a,
        "equal_heat": {c: {k: truth[c][k] for k in ("heater_power_fluid_1601", "energy_balance_rel_err_1601", "sink_fraction_realised_1601")}
                       for c in EQ_CONDITIONS},
        "equal_heat_power_rel_err": {c: truth[c]["heater_power_fluid_1601"] / H_open - 1 for c in EQ_CONDITIONS},
        "layout_C": layout_c.tolist(), "layout_C_in_fluid_all_conditions": True,
        "noise_shape": list(z.shape), "noise_mean": float(z.mean()), "noise_std": float(z.std()),
    }
    ver["passed"] = {
        "layout_A_reproduces_existing_csv": all(same_a.values()),
        "equal_heat_power_within_1e-12": all(abs(v) < 1e-12 for v in ver["equal_heat_power_rel_err"].values()),
        "energy_lt_1e-8": all(truth[c]["energy_balance_rel_err_1601"] < 1e-8 for c in EQ_CONDITIONS),
        "sink_fraction_exact": abs(truth["wall_sink_eq"]["sink_fraction_realised_1601"] - SINK_FRACTION) < 1e-12,
        "layout_C_has_10_distinct_spaced": len({tuple(p) for p in layout_c.tolist()}) == 10,
    }
    (out / "verification_stage1.json").write_text(json.dumps(ver, indent=2))
    return out, config, ver


def write_confirmation_noise(output_dir=DEFAULT_OUTPUT):
    """Add noise_confirm_standard_normal.npz (30 new draws) to an existing stage-1 bundle; never overwrite."""
    path = Path(output_dir) / "noise_confirm_standard_normal.npz"
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    z = np.stack([np.random.default_rng(seed).standard_normal(10) for seed in CONFIRM_SEEDS])
    np.savez(path, z=z, seeds=np.array(CONFIRM_SEEDS))
    return path, z


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--confirm-noise-only", action="store_true",
                   help="only add the 30-draw confirmation noise table to an existing bundle")
    args = p.parse_args(argv)
    if args.confirm_noise_only:
        path, z = write_confirmation_noise(args.output_dir)
        print(json.dumps({"written": str(path), "shape": list(z.shape), "seeds": [CONFIRM_SEEDS[0], CONFIRM_SEEDS[-1]],
                          "mean": float(z.mean()), "std": float(z.std())}))
        return
    out, config, ver = generate(args.output_dir)
    print(json.dumps({"output": str(out), "equal_heat_scale": config["equal_heat_scale"], "verification": ver}, indent=1))
    if not all(ver["passed"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
