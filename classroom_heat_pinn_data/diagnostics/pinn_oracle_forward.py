"""Oracle forward diagnostic for E5b, not an inverse estimator or warm start.

True Bi and f are explicitly supplied before fitting; sensor temperatures are
never used for fitting. See LOG-062. Never overwrite the failed inverse runs.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from classroom_heat_pinn_data.src import common, pinn, run_experiments as rx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    known = common.load_known_e5b()
    # Oracle input, not hidden access by an inverse method.
    cfg = json.loads((common.E5B_DIR / "config_e5b.json").read_text())
    (out / "truth_access.log").write_text("oracle forward INPUT: config_e5b.json, true Bi and f\n")
    f = cfg["sink"]["fraction_of_heater"]
    exp = {**pinn.DEFAULT_EXPERIMENT, **rx.E2_EXPERIMENT, **rx.E2_CONFIGS["C1"], "seed": 0}
    if args.smoke:
        exp.update(adam_steps=3, lbfgs_iters=0, n_interior=40, n_wall=8, hidden=8, depth=1, log_every=1)
    torch.set_num_threads(8)
    torch.manual_seed(0)
    model = pinn.HeatPINN(known, hidden=exp["hidden"], depth=exp["depth"],
                          bi_fixed=cfg["Bi_true"], sink=f)
    assert not model.inverse and not model.learn_sink
    assert all(name.startswith("net.") for name, _ in model.named_parameters())
    assert math.isclose(float(model.sink_fraction()), f, rel_tol=0, abs_tol=1e-14)
    before = model.bi().detach().clone()
    source_hash = hashlib.sha256((ROOT / "classroom_heat_pinn_data/src/pinn.py").read_bytes()).hexdigest()
    def callback(rec):
        if rec["step"] % 1000 == 0 or rec["stage"] == "lbfgs":
            line = json.dumps(rec)
            print(line, flush=True)
            with (out / "progress.jsonl").open("a") as log:
                log.write(line + "\n")
    history, status = pinn.train(model, known, exp, obs=None, on_log=callback)
    torch.testing.assert_close(model.bi(), before, atol=0, rtol=0)
    assert math.isclose(float(model.sink_fraction()), f, rel_tol=0, abs_tol=1e-14)
    assert hashlib.sha256((ROOT / "classroom_heat_pinn_data/src/pinn.py").read_bytes()).hexdigest() == source_hash
    pinn.save_checkpoint(out / "checkpoint.pt", model, exp, history, status,
                         extra={"diagnostic_oracle": True, "sink": f, "not_for_inverse_warm_start": True})
    (out / "history.json").write_text(json.dumps(common.to_jsonable(history), indent=1))
    with (out / "truth_access.log").open("a") as log:
        log.write("post-fit evaluation: reference_e5b.npz\n")
    with np.load(common.E5B_DIR / "reference_e5b.npz") as data:
        ref = {k: data[k] for k in data.files}
    pred = pinn.predict_grid(model, ref["x"], ref["y"])
    diff = pred - ref["theta"]
    relative = float(np.linalg.norm(diff) / np.linalg.norm(ref["theta"]))
    rmse = float(20 * np.sqrt(np.mean(diff ** 2)))
    flows = pinn.heat_flows(model, known)
    targets = {"relative_L2_max": .01, "energy_error_max": .01}
    met = relative <= .01 and flows["energy_balance_relerr"] <= .01
    result = {"purpose": "oracle forward diagnostic, not inverse recovery", "smoke": args.smoke,
              "source_sha256": source_hash, "experiment": exp, "training_status": status,
              "oracle_Bi": cfg["Bi_true"], "oracle_f": f, "relative_L2": relative,
              "temperature_rmse_C": rmse, "heat_flows": flows,
              "targets": targets, "targets_met": met, "fixed_coefficients_preserved": True,
              "sensor_data_used_for_training": False}
    (out / "summary.json").write_text(json.dumps(common.to_jsonable(result), indent=2))
    with (out / "metrics.csv").open("w", newline="") as file:
        columns = ["oracle_f", "relative_L2", "temperature_rmse_C", "energy_error", "targets_met"]
        writer = csv.DictWriter(file, columns)
        writer.writeheader()
        writer.writerow({"oracle_f": f, "relative_L2": relative, "temperature_rmse_C": rmse,
                         "energy_error": flows["energy_balance_relerr"], "targets_met": met})
    print("RESULT " + json.dumps(common.to_jsonable(result)), flush=True)
    return 0 if status["status"] == "completed" and (args.smoke or met) else 2


if __name__ == "__main__":
    raise SystemExit(main())
