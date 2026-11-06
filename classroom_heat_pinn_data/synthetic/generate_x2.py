"""Generate the X2 combined dataset: the X1 internal wall plus the E5b cooling sink (LOG-088).

Same as X1 (k = 1, Bi, heater amplitude, corridor temperature, wall x in [0, 0.6],
y in [0.30, 0.34]) plus the E5b Gaussian sink at (0.75, 0.03), s = 0.05, removing 10 % of
the heater power over the fluid region (amplitude normalised on each grid's fluid control
volumes, as in fd.solve_field). Truth solved on nx = 1601, stored at the 201-grid nodes.

Outputs (in synthetic/x2 by default; an existing directory with files is refused):
  known_x2.json            base known + wall geometry + sink position/width (Bi, strength unknown)
  config_x2.json           ground truth; evaluation only
  reference_x2.npz         1601 truth at 201-grid nodes + fluid / wall-tip masks + 801 field; evaluation only
  sensors_A_x2_clean.csv   layout A, no noise
  sensors_A_x2_noisy.csv   layout A, + N(0, 0.1 C), seed 2029
  verification_x2.json     integrity checks (LOG-088)
"""
import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .generate_data import (BI_TRUE, DT, HEATER, LAYOUT_A, NX_COARSE, NX_FINE, SENSOR_NOISE_C, THETA_INF,
                                T_OUT_C, on_grid_index, to_C, write_sensors)
    from .generate_e5b import SINK, SINK_FRACTION
    from .generate_x1 import NX_CHECK, NX_TRUTH, TIP_RADIUS, WALL, WALL_TIPS, tip_mask
    from .solver import heater_source, make_grid
    from . import solver_wall
else:
    from generate_data import (BI_TRUE, DT, HEATER, LAYOUT_A, NX_COARSE, NX_FINE, SENSOR_NOISE_C, THETA_INF,
                               T_OUT_C, on_grid_index, to_C, write_sensors)
    from generate_e5b import SINK, SINK_FRACTION
    from generate_x1 import NX_CHECK, NX_TRUTH, TIP_RADIUS, WALL, WALL_TIPS, tip_mask
    from solver import heater_source, make_grid
    import solver_wall

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "x2"
BASE_KNOWN = HERE / "data" / "known.json"
SEED_NOISE_X2 = 2029


def source_x2(nx):
    """Heater minus sink on an nx grid; the sink removes SINK_FRACTION of the fluid heater power."""
    x, y, _ = make_grid(nx)
    X, Y = np.meshgrid(x, y, indexing="ij")
    vol = solver_wall.fluid_volumes(x, y, [WALL])
    q_heat = heater_source(X, Y, [HEATER])
    shape = np.exp(-((X - SINK["x"]) ** 2 + (Y - SINK["y"]) ** 2) / (2 * SINK["s"] ** 2))
    heater_power = float((q_heat * vol).sum())
    amp = SINK_FRACTION * heater_power / float((shape * vol).sum())
    return q_heat - amp * shape, heater_power, amp


def solve_x2(nx):
    Q, heater_power, amp = source_x2(nx)
    x, y, theta, info = solver_wall.solve(nx, BI_TRUE, THETA_INF, solids=[WALL], Q=Q)
    return x, y, theta, info, heater_power, amp


def generate(output_dir=DEFAULT_OUTPUT):
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already holds files; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)

    x, y, theta, info, heater_power, amp = solve_x2(NX_TRUTH)
    _, _, _, info_c, heater_c, _ = solve_x2(NX_COARSE)
    x8, y8, theta8, _, _, _ = solve_x2(NX_CHECK)

    s_t, s_8 = (NX_TRUTH - 1) // (NX_FINE - 1), (NX_CHECK - 1) // (NX_FINE - 1)
    xr, yr = x[::s_t], y[::s_t]
    assert np.allclose(xr, make_grid(NX_FINE)[0]) and np.allclose(x8[::s_8], xr)
    th_r, th8_r = theta[::s_t, ::s_t], theta8[::s_8, ::s_8]
    fluid_r = info["active"][::s_t, ::s_t]
    tips_r = tip_mask(xr, yr)
    np.savez(out / "reference_x2.npz", x=xr, y=yr, theta=th_r, T_C=to_C(th_r), fluid=fluid_r,
             wall_tip=tips_r, theta_801=th8_r, Q=info["Q"][::s_t, ::s_t] * fluid_r)

    ix, iy = on_grid_index(x, y, LAYOUT_A)
    clean = theta[ix, iy]
    rng = np.random.default_rng(SEED_NOISE_X2)
    write_sensors(out / "sensors_A_x2_clean.csv", LAYOUT_A, clean)
    write_sensors(out / "sensors_A_x2_noisy.csv", LAYOUT_A,
                  clean + rng.normal(0, SENSOR_NOISE_C / DT, len(LAYOUT_A)))

    known = json.loads(BASE_KNOWN.read_text())
    known["_note"] = ("X2 combined: what estimation may use. Base known.json plus the internal wall geometry "
                      "and the cooling sink's position and width. Bi and sink strength are unknown.")
    known["solids"] = [{**WALL, "type": "adiabatic internal wall"}]
    known["sink"] = {"x": SINK["x"], "y": SINK["y"], "s": SINK["s"],
                     "strength": "unknown: fraction of the fluid heater power removed (to estimate)"}
    (out / "known_x2.json").write_text(json.dumps(known, indent=2))

    sink_power = heater_power - info["q_in"]
    flows = {w: info["loss"][w] for w in "SEN"}
    config = {
        "_note": "X2 ground truth. Generation / evaluation only; estimation code must not read this file.",
        "Bi_true": BI_TRUE, "theta_inf": THETA_INF, "heater": HEATER, "solids": [WALL],
        "sink": {**SINK, "fraction_of_heater": SINK_FRACTION, "amplitude_1601": amp},
        "heater_power_fluid": heater_power,
        "wall_flow_share_of_heater_true": {w: flows[w] / heater_power for w in "SEN"},
        "sink_share_of_heater_true": SINK_FRACTION,
        "noise_C": SENSOR_NOISE_C, "seed_noise": SEED_NOISE_X2,
        "grids": {"truth": NX_TRUTH, "reference_saved": NX_FINE, "uncertainty_check": NX_CHECK},
        "wall_tip_mask": {"tips": WALL_TIPS, "radius": TIP_RADIUS},
        "T_out_C": T_OUT_C, "DT_K": DT, "preregistration": "LOG-088",
    }
    (out / "config_x2.json").write_text(json.dumps(config, indent=2))

    i8x, i8y = on_grid_index(x8, y8, LAYOUT_A)
    diff = np.abs(th_r - th8_r)[fluid_r] * DT
    csv = {k: np.loadtxt(out / f"sensors_A_x2_{k}.csv", delimiter=",", skiprows=1) for k in ("clean", "noisy")}
    ver = {
        "energy_balance_rel_err_1601": info["balance_rel_err"],
        "energy_balance_rel_err_101": info_c["balance_rel_err"],
        "sink_fraction_realised_1601": sink_power / heater_power,
        "sink_fraction_realised_101": (heater_c - info_c["q_in"]) / heater_c,
        "walls_plus_sink_over_heater": (sum(flows.values()) + sink_power) / heater_power,
        "sensors_fluid_and_finite": bool(info["active"][ix, iy].all() and np.isfinite(clean).all()),
        "csv_T_vs_DT_theta_maxdiff_C": max(float(np.abs(d[:, 4] - (T_OUT_C + DT * d[:, 3])).max())
                                           for d in csv.values()),
        "csv_clean_vs_truth_maxdiff_C": float(DT * np.abs(csv["clean"][:, 3] - clean).max()),
        "sensor_801_vs_1601_maxdiff_C": float(DT * np.abs(theta8[i8x, i8y] - clean).max()),
        "field_801_vs_1601_on_201_nodes_C": {
            "max": float(diff.max()), "rms": float(np.sqrt(np.mean(diff ** 2))),
            "max_outside_wall_tip": float((np.abs(th_r - th8_r) * DT)[fluid_r & ~tips_r].max())},
        "min_theta": float(np.nanmin(theta)), "max_theta": float(np.nanmax(theta)),
    }
    ver["passed"] = {
        "energy_lt_1e-8": max(ver["energy_balance_rel_err_1601"], ver["energy_balance_rel_err_101"]) < 1e-8,
        "sink_fraction_exact": abs(ver["sink_fraction_realised_1601"] - SINK_FRACTION) < 1e-12,
        "sensors_fluid_and_finite": ver["sensors_fluid_and_finite"],
        "csv_units_within_1e-4C": ver["csv_T_vs_DT_theta_maxdiff_C"] <= 1e-4
        and ver["csv_clean_vs_truth_maxdiff_C"] <= 1e-4,
        "sensor_801_vs_1601_le_0.005C": ver["sensor_801_vs_1601_maxdiff_C"] <= 0.005,
    }
    (out / "verification_x2.json").write_text(json.dumps(ver, indent=2))
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
