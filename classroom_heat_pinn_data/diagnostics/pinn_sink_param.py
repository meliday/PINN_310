"""E5b development comparison of sink-strength conditioning (pre-registered in LOG-071).

V1 linear_f: f itself is trained (f = gamma) instead of f = exp(gamma), so dr/df = A does not
vanish as f -> 0. V2 data400_sink_sampling: Codex's lambda_data=400 + sink-focused interior
points. V3 combines both. Promotion uses only observations and the FD solver (no truth):
the PINN's Bi and f are re-solved with FD and the sensor RMS of that field is compared.
Reference files are read only after fitting. Run from the repository root.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from classroom_heat_pinn_data.src import common, fd, pinn, run_experiments as rx
from classroom_heat_pinn_data.diagnostics import pinn_sink_diagnosis as diag  # also installs its audit hook

DT = pinn.DTYPE
VARIANTS = ("linear_f", "data400_sink_sampling", "linear_f_data400_sink_sampling")
BASELINE_ADAM5000 = common.RESULTS_DIR / "e5b/codex_diagnosis_baseline_seed0/checkpoint_adam5000.pt"
PROMOTE_FACTOR = 0.5  # FD re-solve sensor RMS must be below this times the baseline Adam5000 value
GATE = {"relerr": 0.05, "q1_rmse_C": 0.02, "nn_fd_field_rms_C": 0.02, "energy_balance_relerr": 0.01}


class LinearSinkPINN(pinn.HeatPINN):
    """Same network and source; gamma holds f itself, not log f."""

    def __init__(self, known, hidden=64, depth=4, sink_init=0.05):
        super().__init__(known, hidden=hidden, depth=depth, sink="learn", sink_init=sink_init)
        with torch.no_grad():
            self.gamma.fill_(sink_init)

    def sink_fraction(self):
        return self.gamma


def make_model(variant, known, exp):
    torch.manual_seed(0)
    cls = LinearSinkPINN if "linear_f" in variant else pinn.HeatPINN
    kwargs = {} if cls is LinearSinkPINN else {"sink": "learn"}
    return cls(known, hidden=exp["hidden"], depth=exp["depth"], sink_init=.05, **kwargs)


def fd_resolve(bi, f, known, obs, nx=fd.INVERSE_NX):
    """FD field for given Bi and f; sensor RMS against the observations (no truth)."""
    x, y, theta, _ = fd.solve_field(bi, known, nx=nx, sink_fraction=f)
    resid = fd.sample(x, y, theta, obs.xy) - obs.theta
    return x, y, theta, float(np.sqrt(np.mean(resid ** 2)) * common.dt_K(known))


def consistency(model, known, obs):
    bi = dict(zip(common.ROBIN_WALLS, model.bi().detach().tolist()))
    f = float(model.sink_fraction().detach())
    x, y, theta, rms = fd_resolve(bi, f, known, obs)
    nn_field = pinn.predict_grid(model, x, y)
    with torch.no_grad():
        nn_sensor = model(torch.tensor(obs.xy, dtype=DT)).reshape(-1).numpy()
    dT = common.dt_K(known)
    return {"Bi": bi, "f": f, "fd_sensor_rms_C": rms,
            "nn_sensor_rms_C": float(np.sqrt(np.mean((nn_sensor - obs.theta) ** 2)) * dT),
            "nn_fd_field_rms_C": float(np.sqrt(np.mean((nn_field - theta) ** 2)) * dT)}


def gate(evaluation, cons):
    checks = {"f": abs(evaluation["f_relerr"]) <= GATE["relerr"],
              **{f"Bi_{w}": abs(evaluation["Bi_relerr"][w]) <= GATE["relerr"] for w in common.ROBIN_WALLS},
              "q1_rmse_C": evaluation["q1_rmse_C"] <= GATE["q1_rmse_C"],
              "nn_fd_field_rms_C": cons["nn_fd_field_rms_C"] <= GATE["nn_fd_field_rms_C"],
              "energy_balance_relerr": evaluation["heat_flows"]["energy_balance_relerr"]
              <= GATE["energy_balance_relerr"]}
    return {"checks": checks, "passed": all(checks.values())}


def train_case(variant, steps, lbfgs, known, obs, out, smoke=False):
    exp = {**pinn.DEFAULT_EXPERIMENT, **rx.E2_EXPERIMENT, **rx.E2_CONFIGS["C1"],
           "seed": 0, "adam_steps": steps, "lbfgs_iters": lbfgs}
    if "data400" in variant:
        exp["lambda_data"] = 400.
    if smoke:
        exp.update(hidden=8, depth=1, n_interior=40, n_wall=8, log_every=1)
    model = make_model(variant, known, exp)
    sampler = diag.Sampler(pinn.sample_interior, focused="sink_sampling" in variant)

    def callback(rec):
        if rec["step"] % 1000 == 0 or rec["stage"] == "lbfgs":
            line = json.dumps(common.to_jsonable({"variant": variant, "data": obs.name, "step": rec["step"],
                              "stage": rec["stage"], "f": rec.get("sink_fraction"),
                              "data_mse": rec["data"], "pde_mse": rec["pde"]}))
            print(line, flush=True)
            with (out / "progress.jsonl").open("a") as fh:
                fh.write(line + "\n")

    started = time.perf_counter()
    diag._FITTING = True  # the diag audit hook blocks truth, checkpoint and summary reads
    try:
        with mock.patch.object(pinn, "sample_interior", sampler):
            history, status = pinn.train(model, known, exp, obs=obs, on_log=callback)
    finally:
        diag._FITTING = False
    tag = f"{variant}_{obs.name}"
    pinn.save_checkpoint(out / f"checkpoint_{tag}.pt", model, exp, history, status,
                         extra={"development": True, "variant": variant, "data": obs.name,
                                "sink_parameter": "linear" if "linear_f" in variant else "exp"})
    diag.write_json(out / f"history_{tag}.json", history)
    cons = consistency(model, known, obs)  # observations + FD only, before any truth read
    result = {"variant": variant, "data": obs.name, "development": True, "experiment": exp,
              "status": status, "wall_time_s": time.perf_counter() - started, "consistency": cons,
              "coverage_mean_1sigma": float(np.mean([c["within_1sigma"] for c in sampler.coverage])),
              "forbidden_fitting_reads": list(diag._FORBIDDEN)}
    result["evaluation"] = diag.score_reference(model, known, obs, out)  # truth, after fitting
    diag.write_json(out / f"result_{tag}.json", result)
    return result


def source_manifest():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(diag.__file__), ROOT / "classroom_heat_pinn_data/src/pinn.py",
                      ROOT / "classroom_heat_pinn_data/src/fd.py",
                      ROOT / "classroom_heat_pinn_data/src/run_experiments.py"]}


def _write_metrics(out, results):
    fields = ["variant", "data", "adam_steps", "status", "f_hat", "f_relerr", "Bi_S_relerr", "Bi_E_relerr",
              "Bi_N_relerr", "q1_rmse_C", "nn_sensor_rms_C", "fd_sensor_rms_C", "nn_fd_field_rms_C",
              "energy_balance_relerr"]
    with (out / "metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        for r in results:
            ev, c = r["evaluation"], r["consistency"]
            w.writerow({"variant": r["variant"], "data": r["data"], "adam_steps": r["experiment"]["adam_steps"],
                        "status": r["status"]["status"], "f_hat": ev["f_hat"], "f_relerr": ev["f_relerr"],
                        **{f"Bi_{k}_relerr": ev["Bi_relerr"][k] for k in common.ROBIN_WALLS},
                        "q1_rmse_C": ev["q1_rmse_C"], "nn_sensor_rms_C": c["nn_sensor_rms_C"],
                        "fd_sensor_rms_C": c["fd_sensor_rms_C"], "nn_fd_field_rms_C": c["nn_fd_field_rms_C"],
                        "energy_balance_relerr": ev["heat_flows"]["energy_balance_relerr"]})


def run(mode, out, variant=None, short_dir=None, clean_dir=None):
    data = "A_sink_noisy" if mode == "noisy" else "A_sink_clean"
    if mode in ("full", "noisy") and variant not in VARIANTS:
        raise ValueError("full/noisy need a preregistered --variant")
    if mode == "full":
        promoted = json.loads((short_dir / "summary.json").read_text())["promoted"]
        if promoted != variant:
            raise ValueError(f"only the promoted variant may run the full budget (promoted: {promoted})")
    if mode == "noisy":
        clean = json.loads((clean_dir / "summary.json").read_text())
        if clean["results"][0]["variant"] != variant or not clean["gate"]["passed"]:
            raise ValueError("noisy runs only for the variant that passed the clean gate")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    before = source_manifest()
    diag.write_json(out / "protocol.json", {
        "mode": mode, "data": data, "seed": 0, "development": True, "preregistration": "LOG-071",
        "variants": VARIANTS, "promotion": f"f>0, completed, FD re-solve sensor RMS < {PROMOTE_FACTOR} x baseline "
        "Adam5000; minimum promoted", "gate": GATE, "source_hashes": before})
    known = common.load_known_e5b()
    obs = common.load_observations(data, known)
    torch.set_num_threads(8)
    summary = {"mode": mode, "preregistration": "LOG-071"}
    if mode == "smoke":
        summary["results"] = [train_case(v, 3, 0, known, obs, out, smoke=True) for v in VARIANTS]
    elif mode == "short":
        baseline = diag.load_model(BASELINE_ADAM5000, known)  # read before any fitting
        summary["baseline_adam5000_consistency"] = consistency(baseline, known, obs)
        summary["results"] = [train_case(v, 5000, 0, known, obs, out) for v in VARIANTS]
        limit = PROMOTE_FACTOR * summary["baseline_adam5000_consistency"]["fd_sensor_rms_C"]
        eligible = [r for r in summary["results"] if r["status"]["status"] == "completed"
                    and r["consistency"]["f"] > 0 and r["consistency"]["fd_sensor_rms_C"] < limit]
        summary["promotion_limit_fd_sensor_rms_C"] = limit
        summary["promoted"] = (min(eligible, key=lambda r: r["consistency"]["fd_sensor_rms_C"])["variant"]
                               if eligible else None)
    else:
        result = train_case(variant, 20000, 100, known, obs, out)
        summary["results"] = [result]
        summary["gate"] = gate(result["evaluation"], result["consistency"])
    assert source_manifest() == before, "source changed during the run"
    diag.write_json(out / "summary.json", summary)
    _write_metrics(out, summary["results"])
    print("SUMMARY " + json.dumps(common.to_jsonable({k: v for k, v in summary.items() if k != "results"})),
          flush=True)
    return summary


class SelfTests(unittest.TestCase):
    def test_linear_parameter_gradient_does_not_scale_with_f(self):
        known = common.load_known_e5b()
        exp = {**pinn.DEFAULT_EXPERIMENT, "hidden": 8, "depth": 1}
        g = torch.Generator().manual_seed(3)
        interior, walls = pinn.sample_interior(200, known, g), pinn.sample_walls(20, known, g)
        for f in (0.05, 1e-4):
            lin, base = make_model("linear_f", known, exp), make_model("data400_sink_sampling", known, exp)
            for a, b in zip(lin.net.parameters(), base.net.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)  # same seed -> same network
            with torch.no_grad():
                lin.gamma.fill_(f)
                base.gamma.fill_(float(np.log(f)))
            torch.testing.assert_close(lin.source(interior, known), base.source(interior, known))
            for m in (lin, base):
                pinn.losses(m, known, interior, walls)[1]["pde"].backward()
            # d/d log f = f * d/df: the exp gradient shrinks with f, the linear one does not.
            self.assertAlmostEqual(float(base.gamma.grad), f * float(lin.gamma.grad), delta=1e-12)

    def test_fd_resolve_reproduces_saved_fd_fit(self):
        known = common.load_known_e5b()
        obs = common.load_observations("A_sink_clean", known)
        saved = json.loads((common.RESULTS_DIR / "e5b/fd_A_sink/summary.json").read_text())
        fit = saved["results"]["A_sink_clean"]["sink"]
        rms = fd_resolve(fit["Bi"], fit["sink_fraction_hat"], known, obs)[3]
        self.assertAlmostEqual(rms, fit["sensor_rms_C"], delta=1e-9)

    def test_full_and_noisy_require_preregistered_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            short = Path(tmp) / "short"
            short.mkdir()
            diag.write_json(short / "summary.json", {"promoted": None})
            with self.assertRaisesRegex(ValueError, "promoted"):
                run("full", Path(tmp) / "full", "linear_f", short_dir=short)
            clean = Path(tmp) / "clean"
            clean.mkdir()
            diag.write_json(clean / "summary.json", {"results": [{"variant": "linear_f"}], "gate": {"passed": False}})
            with self.assertRaisesRegex(ValueError, "clean gate"):
                run("noisy", Path(tmp) / "noisy", "linear_f", clean_dir=clean)
            self.assertFalse((Path(tmp) / "full").exists() or (Path(tmp) / "noisy").exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("self-test", "smoke", "short", "full", "noisy"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--short-dir", type=Path)
    parser.add_argument("--clean-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "self-test":
        res = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SelfTests))
        raise SystemExit(0 if res.wasSuccessful() else 1)
    if args.output_dir is None:
        parser.error("--output-dir is required")
    run(args.mode, args.output_dir, args.variant, args.short_dir, args.clean_dir)


if __name__ == "__main__":
    main()
