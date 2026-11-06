"""Development diagnostics for E5b; never replace the original failed runs.

Run from the repository root with the DeepStudy Python interpreter.
The baseline and all interventions start from fresh seed 0, not FD/truth weights.
Reference files are used only after fitting. See LOG-058 for the fixed protocol.
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
from classroom_heat_pinn_data.src import common, pinn, run_experiments as rx

DT = pinn.DTYPE
VARIANTS = ("data400", "sink_sampling", "freeze2000")
PROBE_STEPS = {0, 100, 1000, 2000, 5000, 10000, 20000}
_FITTING = False
_FORBIDDEN = []


def _audit(event, args):
    if not _FITTING or event != "open" or not isinstance(args[0], (str, bytes)):
        return
    path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
    flags = args[2]
    if not isinstance(flags, int) or flags & 3:  # allow writes, including new checkpoints
        return
    if path.name in common.TRUTH_FILES | common.E5B_TRUTH_FILES or path.suffix == ".pt":
        _FORBIDDEN.append(str(path))
        raise AssertionError(f"forbidden read while fitting: {path}")
    if ROOT / "classroom_heat_pinn_data/results" in path.parents and path.name == "summary.json":
        _FORBIDDEN.append(str(path))
        raise AssertionError(f"result reuse while fitting: {path}")


sys.addaudithook(_audit)


def write_json(path, value):
    path.write_text(json.dumps(common.to_jsonable(value), indent=2))


def make_probes(known):
    g = torch.Generator().manual_seed(20261107)
    interior = pinn.sample_interior(1000, known, g)
    walls = pinn.sample_walls(100, known, g)
    x, y = (np.arange(64) + .5) / 64, (np.arange(52) + .5) * .8 / 52
    X, Y = np.meshgrid(x, y, indexing="ij")
    uniform = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=DT)
    return interior, walls, uniform


def field_checks(model, known, obs, probes):
    _, walls, uniform = probes
    xy = torch.tensor(obs.xy, dtype=DT)
    target = torch.tensor(obs.theta, dtype=DT)
    data = float(((model(xy) - target) ** 2).mean().detach())
    pde = float((pinn.pde_residual(model, uniform, known) ** 2).mean().detach())
    bc = float(torch.stack([(r ** 2).mean() for r in pinn.bc_residuals(model, walls).values()]).mean().detach())
    return {"sensor_theta_mse": data, "sensor_rms_C": common.dt_K(known) * np.sqrt(data),
            "uniform_pde_mse": pde, "probe_bc_mse": bc,
            "selection_score": 400 * data + pde + bc}


def loss_probe(model, known, obs, probes, exp):
    interior, walls, _ = probes
    xy = torch.tensor(obs.xy, dtype=DT)
    target = torch.tensor(obs.theta, dtype=DT)
    _, parts = pinn.losses(model, known, interior, walls, xy, target, exp)
    params = list(model.net.parameters())
    vectors = {}
    for key in ("data", "pde", "bc"):
        grads = torch.autograd.grad(parts[key], params, retain_graph=True, allow_unused=True)
        vectors[key] = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                                  for p, g in zip(params, grads)]).detach()
    gd, gp = vectors["data"], vectors["pde"]
    cosine = float(torch.dot(gd, gp) / (gd.norm() * gp.norm()).clamp_min(1e-30))
    _, _, lap = pinn.derivatives(model, interior)
    geom = model.sink_geom
    shape = torch.exp(-((interior[:, 0] - geom["x"]) ** 2 +
                        (interior[:, 1] - geom["y"]) ** 2) / (2 * geom["s"] ** 2))
    A = (model.heater_power / model.sink_shape_integral * shape).detach()
    r0 = (-lap - pinn.source(interior, known)).detach()
    f = float(model.sink_fraction().detach())
    f_opt = float(-(r0 * A).mean() / (A ** 2).mean())
    gamma_gradient = float(2 * ((r0 + f * A) * A).mean() * f)
    values = {k: float(parts[k].detach()) for k in ("data", "pde", "bc")}
    return {"raw_losses": values,
            "net_gradient_norm": {k: float(v.norm()) for k, v in vectors.items()},
            "weighted_net_gradient_norm": {k: float(vectors[k].norm()) * exp[f"lambda_{k}"]
                                           for k in vectors},
            "data_pde_net_gradient_cosine": cosine,
            "gamma_pde_gradient": gamma_gradient,
            "gamma_data_direct_gradient": 0.0,
            "gamma_active": model.gamma.requires_grad,
            "fixed_network_f_opt_unconstrained": f_opt,
            "fixed_network_f_opt_nonnegative": max(f_opt, 0.0),
            "pde_at_current_f": float(((r0 + f * A) ** 2).mean()),
            "pde_at_zero_f": float((r0 ** 2).mean()),
            "field_checks": field_checks(model, known, obs, probes)}


class Sampler:
    """Keep the base sampler's RNG stream, including subsequent wall draws."""
    def __init__(self, original, focused=False):
        self.original, self.focused = original, focused
        self.calls, self.coverage = 0, []

    def __call__(self, n, known, gen, heater_frac=.3):
        points = self.original(n, known, gen, heater_frac)
        geom = common.sink_geometry(known)
        if self.focused:
            count = int(round(.3 * n))
            if count > n - int(round(heater_frac * n)):
                raise ValueError("not enough uniform points to replace")
            # This extra generator does not change the original interior/wall RNG.
            extra = torch.Generator().manual_seed(731000 + self.calls)
            chunks = []
            while sum(len(a) for a in chunks) < count:
                a = torch.randn(max(2 * count, 1), 2, generator=extra, dtype=DT) * geom["s"]
                a += torch.tensor([geom["x"], geom["y"]], dtype=DT)
                keep = (a[:, 0] > 0) & (a[:, 0] < 1) & (a[:, 1] > 0) & (a[:, 1] < .8)
                chunks.append(a[keep])
            points[:count] = torch.cat(chunks)[:count]
        distance = ((points - torch.tensor([geom["x"], geom["y"]], dtype=DT)) ** 2).sum(1)
        self.coverage.append({"resample": self.calls, "within_1sigma": int((distance < geom["s"] ** 2).sum()),
                              "within_2sigma": int((distance < (2 * geom["s"]) ** 2).sum()), "n": n})
        self.calls += 1
        return points


def load_model(path, known):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    exp = ck["experiment"]
    model = pinn.HeatPINN(known, hidden=exp["hidden"], depth=exp["depth"], sink="learn")
    model.load_state_dict(ck["state_dict"])
    return model


def score_reference(model, known, obs, out):
    # This function is called only after train() has returned.
    with (out / "truth_access.log").open("a") as log:
        log.write(f"post-fit evaluation\t{obs.name}\tconfig_e5b.json\n")
        log.write(f"post-fit evaluation\t{obs.name}\treference_e5b.npz\n")
    cfg = json.loads((common.E5B_DIR / "config_e5b.json").read_text())
    with np.load(common.E5B_DIR / "reference_e5b.npz") as f:
        ref = {k: f[k] for k in f.files}
    pred = pinn.predict_grid(model, ref["x"], ref["y"])
    mask = np.ones(pred.shape, dtype=bool)
    ix = np.rint(obs.xy[:, 0] / (ref["x"][1] - ref["x"][0])).astype(int)
    iy = np.rint(obs.xy[:, 1] / (ref["y"][1] - ref["y"][0])).astype(int)
    mask[ix, iy] = False
    error = (pred - ref["theta"]) * common.dt_K(known)
    f_hat = float(model.sink_fraction().detach())
    bi = dict(zip(common.ROBIN_WALLS, model.bi().detach().tolist()))
    return {"q1_rmse_C": float(np.sqrt(np.mean(error[mask] ** 2))),
            "q1_maxabs_C": float(np.abs(error[mask]).max()),
            "f_hat": f_hat, "f_true": cfg["sink"]["fraction_of_heater"],
            "f_relerr": f_hat / cfg["sink"]["fraction_of_heater"] - 1,
            "Bi": bi, "Bi_relerr": {w: bi[w] / cfg["Bi_true"][w] - 1 for w in bi},
            "heat_flows": pinn.heat_flows(model, known)}


def train_case(variant, steps, lbfgs, known, obs, out, smoke=False):
    global _FITTING
    exp = {**pinn.DEFAULT_EXPERIMENT, **rx.E2_EXPERIMENT, **rx.E2_CONFIGS["C1"],
           "seed": 0, "adam_steps": steps, "lbfgs_iters": lbfgs}
    if variant == "data400":
        exp["lambda_data"] = 400.
    if smoke:
        exp.update(hidden=8, depth=1, n_interior=40, n_wall=8, log_every=1)
    probes = make_probes(known)
    torch.manual_seed(0)
    model = pinn.HeatPINN(known, hidden=exp["hidden"], depth=exp["depth"], sink="learn", sink_init=.05)
    if variant == "freeze2000":
        model.gamma.requires_grad_(False)
    sampler = Sampler(pinn.sample_interior, focused=variant == "sink_sampling")
    probes_log = []

    def progress(value):
        line = json.dumps(common.to_jsonable(value))
        print(line, flush=True)
        with (out / "progress.jsonl").open("a") as f:
            f.write(line + "\n")

    def callback(rec):
        if rec["step"] in PROBE_STEPS or rec["stage"] == "lbfgs":
            probe = {"step": rec["step"], "stage": rec["stage"],
                     "f": float(model.sink_fraction().detach()), **loss_probe(model, known, obs, probes, exp)}
            probes_log.append(probe)
            write_json(out / f"probes_{variant}.json", probes_log)
        if variant == "freeze2000" and rec["step"] >= 2000 and not model.gamma.requires_grad:
            model.gamma.requires_grad_(True)
        if variant == "baseline" and rec["step"] == 5000 and rec["stage"] == "adam":
            torch.save({"state_dict": model.state_dict(), "experiment": exp},
                       out / "checkpoint_adam5000.pt")
        if rec["step"] % 1000 == 0 or rec["stage"] == "lbfgs":
            progress({"variant": variant, "data": obs.name, "step": rec["step"], "stage": rec["stage"],
                      "f": rec.get("sink_fraction"), "data_mse": rec["data"], "pde_mse": rec["pde"]})

    started = time.perf_counter()
    _FITTING = True
    try:
        with mock.patch.object(pinn, "sample_interior", sampler):
            history, status = pinn.train(model, known, exp, obs=obs, on_log=callback)
    finally:
        _FITTING = False
    path = out / f"checkpoint_{variant}.pt"
    pinn.save_checkpoint(path, model, exp, history, status,
                         extra={"development": True, "variant": variant, "data": obs.name, "sink": "learn"})
    write_json(out / f"history_{variant}.json", history)
    write_json(out / f"coverage_{variant}.json", sampler.coverage)
    checks = field_checks(model, known, obs, probes)
    result = {"variant": variant, "data": obs.name, "development": True, "experiment": exp,
              "status": status, "wall_time_with_diagnostics_s": time.perf_counter() - started,
              "checks": checks, "evaluation": score_reference(model, known, obs, out),
              "coverage_mean_1sigma": float(np.mean([c["within_1sigma"] for c in sampler.coverage])),
              "coverage_mean_2sigma": float(np.mean([c["within_2sigma"] for c in sampler.coverage])),
              "forbidden_fitting_reads": list(_FORBIDDEN)}
    write_json(out / f"result_{variant}.json", result)
    return result, model


def source_manifest():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT / "classroom_heat_pinn_data/src/pinn.py", Path(__file__),
                      ROOT / "classroom_heat_pinn_data/src/run_experiments.py"]}


def run(mode, out, baseline_dir=None, variant=None, data="A_sink_clean"):
    if mode == "baseline" and data != "A_sink_clean":
        raise ValueError("baseline reproduction is defined for A_sink_clean")
    baseline = None
    if mode == "ablations":
        if baseline_dir is None:
            raise ValueError("ablations require --baseline-dir")
        saved = json.loads((baseline_dir / "summary.json").read_text())
        if saved["results"][0]["data"] != data:
            raise ValueError("ablation data must match the baseline control")
        baseline = saved["control_adam5000"]
    if mode == "promote" and variant not in VARIANTS:
        raise ValueError("promote requires a preregistered variant")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    before = source_manifest()
    write_json(out / "protocol.json", {"mode": mode, "data": data, "seed": 0, "development": True,
               "source_hashes": before, "reference_selection": False,
               "selection": "minimum finite 400*sensor_theta_mse+uniform_pde_mse+bc_mse below baseline Adam5000",
               "interventions": {"data400": "lambda_data only", "sink_sampling": "interior mixture only",
                                 "freeze2000": "freeze gamma for 2000 Adam steps only"}})
    known = common.load_known_e5b()
    obs = common.load_observations(data, known)
    torch.set_num_threads(8)
    if mode in ("baseline", "smoke", "promote"):
        case = variant if mode == "promote" else "baseline"
        result, model = train_case(case, 3 if mode == "smoke" else 20000,
                                  0 if mode == "smoke" else 100, known, obs, out, smoke=mode == "smoke")
        summary = {"mode": mode, "results": [result]}
        if mode == "baseline":
            saved = load_model(common.RESULTS_DIR / "e5b/pinn_A_sink_seed0/checkpoint_A_sink_clean.pt", known)
            difference = max(float((model.state_dict()[k] - saved.state_dict()[k]).abs().max())
                             for k in model.state_dict())
            summary["reproduction_max_parameter_difference"] = difference
            control = load_model(out / "checkpoint_adam5000.pt", known)
            summary["control_adam5000"] = {"checks": field_checks(control, known, obs, make_probes(known)),
                                          "evaluation": score_reference(control, known, obs, out)}
    else:
        results = []
        for case in VARIANTS:
            result, _ = train_case(case, 5000, 0, known, obs, out)
            results.append(result)
        eligible = [r for r in results if r["status"]["status"] == "completed"
                    and np.isfinite(r["checks"]["selection_score"])
                    and r["checks"]["selection_score"] < baseline["checks"]["selection_score"]]
        chosen = min(eligible, key=lambda r: r["checks"]["selection_score"])["variant"] if eligible else None
        summary = {"mode": mode, "results": results, "baseline_adam5000": baseline,
                   "selected_by_training_probes": chosen}
    assert source_manifest() == before, "source changed during diagnostic run"
    write_json(out / "summary.json", summary)
    with (out / "metrics.csv").open("w", newline="") as f:
        fields = ["variant", "data", "adam_steps", "f_hat", "f_relerr", "q1_rmse_C",
                  "sensor_rms_C", "uniform_pde_mse", "selection_score"]
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for r in summary["results"]:
            writer.writerow({"variant": r["variant"], "data": r["data"],
                "adam_steps": r["experiment"]["adam_steps"],
                **{k: r["evaluation"][k] for k in ("f_hat", "f_relerr", "q1_rmse_C")},
                **{k: r["checks"][k] for k in ("sensor_rms_C", "uniform_pde_mse", "selection_score")}})
    print("RESULT " + json.dumps(common.to_jsonable(summary)), flush=True)
    return summary


class DiagnosticChecks(unittest.TestCase):
    def test_incompatible_data_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with self.assertRaisesRegex(ValueError, "A_sink_clean"):
                run("baseline", out, data="A_sink_noisy")
            baseline = Path(tmp) / "baseline"
            baseline.mkdir()
            write_json(baseline / "summary.json",
                       {"results": [{"data": "A_sink_clean"}], "control_adam5000": {}})
            with self.assertRaisesRegex(ValueError, "must match"):
                run("ablations", out, baseline, data="A_sink_noisy")
            self.assertFalse(out.exists())

    def test_profile_matches_loss_gradient_without_mutating_optimizer_gradients(self):
        known = common.load_known_e5b()
        obs = common.load_observations("A_sink_clean", known)
        torch.manual_seed(0)
        model = pinn.HeatPINN(known, hidden=8, depth=1, sink="learn")
        probes = make_probes(known)
        total, parts = pinn.losses(model, known, probes[0], probes[1],
                                  torch.tensor(obs.xy, dtype=DT), torch.tensor(obs.theta, dtype=DT))
        total.backward()
        before = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
        rec = loss_probe(model, known, obs, probes, pinn.DEFAULT_EXPERIMENT)
        self.assertAlmostEqual(rec["gamma_pde_gradient"], float(model.gamma.grad), places=10)
        self.assertAlmostEqual(rec["pde_at_current_f"], float(parts["pde"].detach()), places=10)
        for param, previous in zip(model.parameters(), before):
            if previous is not None:
                torch.testing.assert_close(param.grad, previous, rtol=0, atol=0)

    def test_sampling_changes_only_interior_mixture_and_preserves_wall_rng(self):
        known = common.load_known_e5b()
        a, b = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
        original = pinn.sample_interior(1000, known, a)
        focused = Sampler(pinn.sample_interior, True)(1000, known, b)
        torch.testing.assert_close(original[300:], focused[300:], rtol=0, atol=0)
        wa, wb = pinn.sample_walls(100, known, a), pinn.sample_walls(100, known, b)
        for wall in wa:
            torch.testing.assert_close(wa[wall], wb[wall], rtol=0, atol=0)
        self.assertEqual(focused.shape, original.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("self-test", "smoke", "baseline", "ablations", "promote"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--data", choices=("A_sink_clean", "A_sink_noisy"), default="A_sink_clean")
    args = parser.parse_args()
    if args.mode == "self-test":
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DiagnosticChecks))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.output_dir is None or (args.mode == "promote" and args.variant is None):
        parser.error("--output-dir is required; promote also needs --variant")
    run(args.mode, args.output_dir, args.baseline_dir, args.variant, args.data)


if __name__ == "__main__":
    main()
