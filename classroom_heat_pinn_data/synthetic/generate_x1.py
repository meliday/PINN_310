"""Generate the X1 dataset: the base room with one internal adiabatic wall (LOG-073, LOG-078).

User-requested extension (plan section 17.3). Same k = 1, Bi, heater (amplitude fixed) and
corridor temperature as the base model; a solid wall x in [0, 0.6], y in [0.30, 0.34] is
attached to the adiabatic W wall, leaving a 0.4-wide passage on the E side. No sink.

The truth field is solved on nx = 1601 (a numerical reference, not an exact solution; the
re-entrant wall tip limits convergence, LOG-074/075). Sensor values are taken at 1601 grid
nodes, no interpolation.

Outputs (in synthetic/x1 by default; an existing directory with files is refused):
  known_x1.json            what estimation may use: base known.json + wall geometry (Bi unknown)
  config_x1.json           ground truth (Bi, heater power in the fluid region, wall shares); evaluation only
  reference_x1.npz         1601 truth at the 201-grid nodes + fluid / wall-tip masks + 801 field; evaluation only
  sensors_A_x1_clean.csv   layout A, no noise
  sensors_A_x1_noisy.csv   layout A, + N(0, 0.1 C), seed 2028
  verification_x1.json     integrity checks (pre-registered in LOG-078)
"""
import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .generate_data import (BI_TRUE, DT, HEATER, LAYOUT_A, NX_COARSE, NX_FINE, SENSOR_NOISE_C, THETA_INF,
                                T_OUT_C, on_grid_index, to_C, write_sensors)
    from .solver import heater_source, make_grid
    from . import solver_wall
else:
    from generate_data import (BI_TRUE, DT, HEATER, LAYOUT_A, NX_COARSE, NX_FINE, SENSOR_NOISE_C, THETA_INF,
                               T_OUT_C, on_grid_index, to_C, write_sensors)
    from solver import heater_source, make_grid
    import solver_wall

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "x1"
BASE_KNOWN = HERE / "data" / "known.json"
WALL = {"x0": 0.0, "x1": 0.60, "y0": 0.30, "y1": 0.34}
WALL_TIPS = ((0.60, 0.30), (0.60, 0.34))
TIP_RADIUS = 0.1
NX_TRUTH, NX_CHECK = 1601, 801
SEED_NOISE_X1 = 2028


def solve_x1(nx):
    return solver_wall.solve(nx, BI_TRUE, THETA_INF, solids=[WALL], heaters=[HEATER])


def tip_mask(x, y):
    X, Y = np.meshgrid(x, y, indexing="ij")
    return np.min([np.hypot(X - a, Y - b) for a, b in WALL_TIPS], axis=0) < TIP_RADIUS


def generate(output_dir=DEFAULT_OUTPUT):
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already holds files; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)

    x, y, theta, info = solve_x1(NX_TRUTH)
    xc, yc, theta_c, info_c = solve_x1(NX_COARSE)
    x8, y8, theta8, _ = solve_x1(NX_CHECK)

    # reference on the 201-grid nodes, taken exactly from the 1601 and 801 nodes
    s_t, s_8 = (NX_TRUTH - 1) // (NX_FINE - 1), (NX_CHECK - 1) // (NX_FINE - 1)
    xr, yr = x[::s_t], y[::s_t]
    assert np.allclose(xr, make_grid(NX_FINE)[0]) and np.allclose(x8[::s_8], xr)
    th_r, th8_r = theta[::s_t, ::s_t], theta8[::s_8, ::s_8]
    fluid_r = info["active"][::s_t, ::s_t]
    tips_r = tip_mask(xr, yr)
    Xr, Yr = np.meshgrid(xr, yr, indexing="ij")
    np.savez(out / "reference_x1.npz", x=xr, y=yr, theta=th_r, T_C=to_C(th_r), fluid=fluid_r,
             wall_tip=tips_r, theta_801=th8_r, Q=heater_source(Xr, Yr, [HEATER]) * fluid_r)

    ix, iy = on_grid_index(x, y, LAYOUT_A)
    clean = theta[ix, iy]
    rng = np.random.default_rng(SEED_NOISE_X1)
    write_sensors(out / "sensors_A_x1_clean.csv", LAYOUT_A, clean)
    write_sensors(out / "sensors_A_x1_noisy.csv", LAYOUT_A,
                  clean + rng.normal(0, SENSOR_NOISE_C / DT, len(LAYOUT_A)))

    known = json.loads(BASE_KNOWN.read_text())
    known["_note"] = ("X1: what estimation may use. Base known.json plus the internal wall geometry. "
                      "Bi are unknown.")
    known["solids"] = [{**WALL, "type": "adiabatic internal wall"}]
    (out / "known_x1.json").write_text(json.dumps(known, indent=2))

    # heater power without the wall (same grid, full control volumes), to report what the solid removes
    X, Y = np.meshgrid(x, y, indexing="ij")
    h = x[1] - x[0]
    wx = np.full(len(x), h); wx[[0, -1]] = h / 2
    wy = np.full(len(y), h); wy[[0, -1]] = h / 2
    heater_no_wall = float((heater_source(X, Y, [HEATER]) * np.outer(wx, wy)).sum())
    heater_power = info["q_in"]
    flows = {w: info["loss"][w] for w in "SEN"}
    config = {
        "_note": "X1 ground truth. Generation / evaluation only; estimation code must not read this file.",
        "Bi_true": BI_TRUE, "theta_inf": THETA_INF, "heater": HEATER, "solids": [WALL],
        "heater_power_fluid": heater_power, "heater_power_no_wall": heater_no_wall,
        "heater_removed_by_wall_fraction": 1 - heater_power / heater_no_wall,
        "wall_flow_share_of_heater_true": {w: flows[w] / heater_power for w in "SEN"},
        "noise_C": SENSOR_NOISE_C, "seed_noise": SEED_NOISE_X1,
        "grids": {"truth": NX_TRUTH, "reference_saved": NX_FINE, "uncertainty_check": NX_CHECK,
                  "inverse_fd_default": NX_COARSE},
        "wall_tip_mask": {"tips": WALL_TIPS, "radius": TIP_RADIUS},
        "T_out_C": T_OUT_C, "DT_K": DT, "preregistration": "LOG-078",
    }
    (out / "config_x1.json").write_text(json.dumps(config, indent=2))

    # ---- verification (criteria fixed in LOG-078)
    i8x, i8y = on_grid_index(x8, y8, LAYOUT_A)
    diff = np.abs(th_r - th8_r)[fluid_r] * DT
    csv = {k: np.loadtxt(out / f"sensors_A_x1_{k}.csv", delimiter=",", skiprows=1) for k in ("clean", "noisy")}
    ver = {
        "energy_balance_rel_err_1601": info["balance_rel_err"],
        "energy_balance_rel_err_101": info_c["balance_rel_err"],
        "sensors_fluid_and_finite": bool(info["active"][ix, iy].all() and np.isfinite(clean).all()),
        "csv_T_vs_DT_theta_maxdiff_C": max(float(np.abs(d[:, 4] - (T_OUT_C + DT * d[:, 3])).max())
                                           for d in csv.values()),
        "csv_clean_vs_truth_maxdiff_C": float(DT * np.abs(csv["clean"][:, 3] - clean).max()),
        "sensor_801_vs_1601_maxdiff_C": float(DT * np.abs(theta8[i8x, i8y] - clean).max()),
        "heater_removed_by_wall_fraction": config["heater_removed_by_wall_fraction"],
        "field_801_vs_1601_on_201_nodes_C": {
            "max": float(diff.max()), "rms": float(np.sqrt(np.mean(diff ** 2))),
            "max_outside_wall_tip": float((np.abs(th_r - th8_r) * DT)[fluid_r & ~tips_r].max())},
        "min_theta": float(np.nanmin(theta)), "max_theta": float(np.nanmax(theta)),
    }
    ver["passed"] = {
        "energy_lt_1e-8": max(ver["energy_balance_rel_err_1601"], ver["energy_balance_rel_err_101"]) < 1e-8,
        "sensors_fluid_and_finite": ver["sensors_fluid_and_finite"],
        "csv_units_within_1e-4C": ver["csv_T_vs_DT_theta_maxdiff_C"] <= 1e-4
        and ver["csv_clean_vs_truth_maxdiff_C"] <= 1e-4,
        "sensor_801_vs_1601_le_0.005C": ver["sensor_801_vs_1601_maxdiff_C"] <= 0.005,
        "heater_removed_in_0.019_0.022": 0.019 <= ver["heater_removed_by_wall_fraction"] <= 0.022,
    }
    (out / "verification_x1.json").write_text(json.dumps(ver, indent=2))
    return out, config, ver


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = p.parse_args(argv)
    out, config, ver = generate(args.output_dir)
    print(json.dumps({"output": str(out), "verification": ver,
                      "wall_flow_share_of_heater_true": config["wall_flow_share_of_heater_true"],
                      "heater_power_fluid": config["heater_power_fluid"]}, indent=2))
    if not all(ver["passed"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
