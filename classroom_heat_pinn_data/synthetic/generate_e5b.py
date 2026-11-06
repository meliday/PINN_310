"""Generate the E5b dataset: the simple model plus ONE known-position cooling sink.

User-requested extension (plan section 17.2). Everything matches the base synthetic
model (k = 1, same Bi, heater and corridor temperature) except a Gaussian heat sink at
the window frame, with the same position, width and strength as the E5 mismatch draft,
so its effect can be studied in isolation (no variable k, no sensor offsets).

Outputs (in synthetic/e5b by default; an existing directory with files is refused):
  known_e5b.json            what estimation may use: base known.json + sink position/width
                            (sink strength is the unknown to be estimated)
  config_e5b.json           ground truth (Bi, sink fraction/amplitude, wall flows); evaluation only
  reference_e5b.npz         true field on the 201 x 161 grid; evaluation only
  sensors_A_sink_clean.csv  layout A, no noise
  sensors_A_sink_noisy.csv  layout A, + N(0, 0.1 C)
  verification_e5b.json     energy balance, grid refinement, sensor and unit checks
"""
import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .generate_data import (BI_TRUE, DRAFT, DRAFT_FRACTION, DT, HEATER, LAYOUT_A, NX_COARSE,
                                NX_FINE, SENSOR_NOISE_C, THETA_INF, T_OUT_C, on_grid_index, to_C,
                                write_sensors)
    from .solver import LX, LY, heater_source, make_grid, solve
else:
    from generate_data import (BI_TRUE, DRAFT, DRAFT_FRACTION, DT, HEATER, LAYOUT_A, NX_COARSE,
                               NX_FINE, SENSOR_NOISE_C, THETA_INF, T_OUT_C, on_grid_index, to_C,
                               write_sensors)
    from solver import LX, LY, heater_source, make_grid, solve

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "e5b"
BASE_KNOWN = HERE / "data" / "known.json"
SINK = dict(DRAFT)               # same position/width as the E5 draft: x=0.75, y=0.03, s=0.05
SINK_FRACTION = DRAFT_FRACTION   # removes 10 % of the heater power
SEED_NOISE_E5B = 2027


def fv_volumes(x, y):
    h = x[1] - x[0]
    wx = np.full(len(x), h); wx[[0, -1]] = h / 2
    wy = np.full(len(y), h); wy[[0, -1]] = h / 2
    return np.outer(wx, wy)


def sink_source(nx):
    """Heater minus sink on an nx grid; the amplitude makes the sink remove SINK_FRACTION
    of the heater power on THAT grid's control volumes."""
    x, y, _ = make_grid(nx)
    X, Y = np.meshgrid(x, y, indexing="ij")
    vol = fv_volumes(x, y)
    q_heat = heater_source(X, Y, [HEATER])
    shape = np.exp(-((X - SINK["x"]) ** 2 + (Y - SINK["y"]) ** 2) / (2 * SINK["s"] ** 2))
    amp = SINK_FRACTION * (q_heat * vol).sum() / (shape * vol).sum()
    return x, y, q_heat - amp * shape, float((q_heat * vol).sum()), float(amp)


def generate(output_dir=DEFAULT_OUTPUT):
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already holds files; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)

    x, y, Q, heater_power, amp = sink_source(NX_FINE)
    _, _, theta, info = solve(NX_FINE, BI_TRUE, THETA_INF, Q=Q)
    np.savez(out / "reference_e5b.npz", x=x, y=y, theta=theta, T_C=to_C(theta), Q=Q)

    ix, iy = on_grid_index(x, y, LAYOUT_A)
    clean = theta[ix, iy]
    rng = np.random.default_rng(SEED_NOISE_E5B)
    write_sensors(out / "sensors_A_sink_clean.csv", LAYOUT_A, clean)
    write_sensors(out / "sensors_A_sink_noisy.csv", LAYOUT_A, clean + rng.normal(0, SENSOR_NOISE_C / DT, len(LAYOUT_A)))

    known = json.loads(BASE_KNOWN.read_text())
    known["_note"] = ("E5b: what estimation may use. Base known.json plus the cooling sink's position and "
                      "width. Sink strength and Bi are unknown.")
    known["sink"] = {"x": SINK["x"], "y": SINK["y"], "s": SINK["s"],
                     "strength": "unknown: fraction of heater power removed (to estimate)"}
    (out / "known_e5b.json").write_text(json.dumps(known, indent=2))

    flows = {w: info["loss"][w] for w in "SEN"}
    config = {
        "_note": "E5b ground truth. Generation / evaluation only; estimation code must not read this file.",
        "Bi_true": BI_TRUE, "theta_inf": THETA_INF, "heater": HEATER,
        "sink": {**SINK, "fraction_of_heater": SINK_FRACTION, "amplitude_201": amp},
        "heater_power": heater_power,
        "wall_flow_share_of_heater_true": {w: flows[w] / heater_power for w in "SEN"},
        "sink_share_of_heater_true": SINK_FRACTION,
        "noise_C": SENSOR_NOISE_C, "seed_noise": SEED_NOISE_E5B,
        "grids": {"reference": [NX_FINE, len(y)], "inverse_fd": NX_COARSE},
        "T_out_C": T_OUT_C, "DT_K": DT,
    }
    (out / "config_e5b.json").write_text(json.dumps(config, indent=2))

    # ---- verification
    xc, yc, Qc, heater_c, _ = sink_source(NX_COARSE)
    _, _, theta_c, info_c = solve(NX_COARSE, BI_TRUE, THETA_INF, Q=Qc)
    sink_power = heater_power - info["q_in"]
    ver = {
        "energy_balance_rel_err_201": info["balance_rel_err"],
        "energy_balance_rel_err_101": info_c["balance_rel_err"],
        "sink_fraction_realised_201": sink_power / heater_power,
        "sink_fraction_realised_101": (heater_c - info_c["q_in"]) / heater_c,
        "wall_plus_sink_over_heater": (sum(flows.values()) + sink_power) / heater_power,
        "refinement_maxdiff_C": float(DT * np.abs(theta[::2, ::2] - theta_c).max()),
        # Read the written CSVs back: rounded T_obs_C must equal T_out + DT * rounded theta_obs.
        "csv_T_vs_DT_theta_maxdiff_C": max(
            float(np.abs(d[:, 4] - (T_OUT_C + DT * d[:, 3])).max())
            for d in (np.loadtxt(out / f"sensors_A_sink_{k}.csv", delimiter=",", skiprows=1)
                      for k in ("clean", "noisy"))),
        "csv_clean_vs_reference_maxdiff_C": float(DT * np.abs(np.loadtxt(
            out / "sensors_A_sink_clean.csv", delimiter=",", skiprows=1)[:, 3] - clean).max()),
        "sensors_on_101_and_201_nodes": on_grid_index(xc, yc, LAYOUT_A) is not None,  # asserts otherwise
        "min_theta": float(theta.min()), "max_theta": float(theta.max()),
    }
    ver["passed"] = {
        "energy_lt_1e-8": max(ver["energy_balance_rel_err_201"], ver["energy_balance_rel_err_101"]) < 1e-8,
        "sink_fraction_exact": abs(ver["sink_fraction_realised_201"] - SINK_FRACTION) < 1e-12,
        "refinement_lt_0.01C": ver["refinement_maxdiff_C"] < 0.01,
        "csv_units_within_1e-4C": ver["csv_T_vs_DT_theta_maxdiff_C"] <= 1e-4,
    }
    (out / "verification_e5b.json").write_text(json.dumps(ver, indent=2))
    return out, config, ver


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = p.parse_args(argv)
    out, config, ver = generate(args.output_dir)
    print(json.dumps({"output": str(out), "verification": ver,
                      "wall_flow_share_of_heater_true": config["wall_flow_share_of_heater_true"],
                      "sink_share_of_heater_true": config["sink_share_of_heater_true"]}, indent=2))
    if not all(ver["passed"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
