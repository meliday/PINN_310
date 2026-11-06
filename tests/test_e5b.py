"""E5b (user-requested extension): loader contract for the sink-only data set."""

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

from contextlib import redirect_stderr
import io
from unittest import mock

from classroom_heat_pinn_data.src import baselines, common, fd, run_experiments as rx

# Records every file opened while armed (same approach as test_e0_baselines).
_OPENED = []
_ARMED = [False]


def _audit(event, args):
    if _ARMED[0] and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        _OPENED.append(os.path.realpath(os.fsdecode(args[0])))


sys.addaudithook(_audit)
E5B_TRUTH_REAL = {os.path.realpath(common.E5B_DIR / f) for f in common.E5B_TRUTH_FILES}


class _Watch:
    def __enter__(self):
        _OPENED.clear()
        _ARMED[0] = True
        return _OPENED

    def __exit__(self, *exc):
        _ARMED[0] = False


class E5bLoaderTests(unittest.TestCase):
    def test_known_gives_sink_geometry_but_not_strength(self):
        known = common.load_known_e5b()
        self.assertEqual(common.sink_geometry(known), {"x": 0.75, "y": 0.03, "s": 0.05})
        self.assertNotIsInstance(known["sink"].get("strength"), (int, float))
        self.assertIsNone(common.sink_geometry(common.load_known()))

    def test_sink_observations_load_from_e5b_dir_without_truth(self):
        with _Watch() as opened:
            known = common.load_known_e5b()
            clean = common.load_observations("A_sink_clean", known)
            noisy = common.load_observations("A_sink_noisy", known)
        self.assertTrue(any(p.endswith("e5b/sensors_A_sink_clean.csv") for p in opened))
        self.assertFalse(set(opened) & E5B_TRUTH_REAL, f"truth opened: {set(opened) & E5B_TRUTH_REAL}")
        base = common.load_observations("A_clean", common.load_known())
        np.testing.assert_array_equal(clean.xy, base.xy)  # same layout A
        self.assertGreater(np.abs(noisy.theta - clean.theta).max(), 0)

    def test_watch_detects_e5b_truth_access(self):
        # Positive control for the hook, through a symlink alias.
        with tempfile.TemporaryDirectory() as tmp:
            alias = Path(tmp) / "alias.json"
            alias.symlink_to(common.E5B_DIR / "config_e5b.json")
            with _Watch() as opened:
                alias.read_text()
        self.assertTrue(set(opened) & E5B_TRUTH_REAL)

    def test_truth_names_are_rejected(self):
        known = common.load_known_e5b()
        for name in ("../e5b/config_e5b", "e5b", "A_sink"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                common.load_observations(name, known)

    def test_known_with_numeric_strength_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            known = json.loads((common.E5B_DIR / "known_e5b.json").read_text())
            known["sink"]["strength"] = 0.1
            (Path(tmp) / "known_e5b.json").write_text(json.dumps(known))
            with self.assertRaises(ValueError):
                common.load_known_e5b(tmp)

    def test_base_loader_still_reads_base_dir(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        shutil.copy(common.DATA_DIR / "sensors_A_clean.csv", tmp)
        known = common.load_known()
        self.assertEqual(common.load_observations("A_clean", known, tmp).xy.shape, (10, 2))


class FdSinkTests(unittest.TestCase):
    BI = (1.0, 0.4, 0.25)

    def setUp(self):
        self.known = common.load_known_e5b()

    def test_zero_sink_equals_base_model(self):
        _, _, base, info_b = fd.solve_field(self.BI, common.load_known())
        _, _, zero, info_z = fd.solve_field(self.BI, self.known, sink_fraction=0.0)
        np.testing.assert_allclose(zero, base, atol=1e-13, rtol=0)
        self.assertAlmostEqual(info_z["sink_power"], 0.0, places=12)
        # Base path unchanged: no sink keys, shares relative to the heater as before.
        self.assertNotIn("share_sink", fd.heat_flows(info_b))

    def test_sink_removes_requested_fraction_and_balances(self):
        for nx in (fd.INVERSE_NX, fd.FINE_NX):
            _, _, _, info = fd.solve_field(self.BI, self.known, nx=nx, sink_fraction=0.1)
            fl = fd.heat_flows(info)
            self.assertAlmostEqual(fl["share_sink"], 0.1, places=12)
            self.assertLess(fl["energy_balance_relerr"], 1e-10)
            # walls (incl. adiabatic W ~ 0) + sink = heater
            self.assertAlmostEqual(sum(fl["share_signed"].values()) + fl["share_sink"], 1.0, places=9)

    def test_sink_cools_nearby_sensor(self):
        pt = np.array([[0.70, 0.04]])
        x, y, t0, _ = fd.solve_field(self.BI, self.known, sink_fraction=0.0)
        _, _, t1, _ = fd.solve_field(self.BI, self.known, sink_fraction=0.1)
        self.assertLess(fd.sample(x, y, t1, pt)[0], fd.sample(x, y, t0, pt)[0])

    def test_sink_requires_geometry(self):
        with self.assertRaises(ValueError):
            fd.solve_field(self.BI, common.load_known(), sink_fraction=0.1)

    def test_reproduces_generated_reference_with_true_parameters(self):
        # Tool diagnostic: truth read only to check the wrapper against the generator.
        cfg = json.loads((common.E5B_DIR / "config_e5b.json").read_text())
        with np.load(common.E5B_DIR / "reference_e5b.npz") as ref:
            theta_ref = ref["theta"]
        _, _, theta, _ = fd.solve_field(cfg["Bi_true"], self.known, nx=fd.FINE_NX,
                                        sink_fraction=cfg["sink"]["fraction_of_heater"])
        self.assertLess(20 * np.abs(theta - theta_ref).max(), 1e-9)


class FdInverseSinkTests(unittest.TestCase):
    def test_clean_recovery_of_bi_and_sink_without_truth_access(self):
        known = common.load_known_e5b()
        with _Watch() as opened:
            obs = common.load_observations("A_sink_clean", known)
            fit = baselines.fit_fd_inverse(obs, known, {"sink": True, "starts": [(1, 1, 1, 0.05)]})
        self.assertFalse(set(opened) & E5B_TRUTH_REAL, "fitting opened E5b truth")
        self.assertEqual(fit["status"], "completed")
        self.assertEqual(fit["jacobian"]["rank"], 4)
        # Truth read only to score the finished fit.
        cfg = json.loads((common.E5B_DIR / "config_e5b.json").read_text())
        for w in common.ROBIN_WALLS:
            self.assertLess(abs(fit["Bi"][w] / cfg["Bi_true"][w] - 1), 5e-3, w)
        self.assertLess(abs(fit["sink_fraction"] / cfg["sink"]["fraction_of_heater"] - 1), 5e-3)

    def test_base_fit_unchanged_and_has_no_sink(self):
        known = common.load_known()
        obs = common.load_observations("A_clean", known)
        fit = baselines.fit_fd_inverse(obs, known, {"starts": [(1, 1, 1)]})
        self.assertIsNone(fit["sink_fraction"])
        self.assertEqual(len(fit["beta"]), 3)
        # Same numbers as the stored E0 result for this start (results/e0/A_clean/fd_inverse_runs.json).
        stored = json.loads((common.RESULTS_DIR / "e0" / "A_clean" / "fd_inverse_runs.json").read_text())
        ref = next(r for r in stored if r["start"] == [1.0, 1.0, 1.0])
        for w in common.ROBIN_WALLS:
            self.assertAlmostEqual(fit["Bi"][w], ref["Bi"][w], places=10)

    def test_sink_bounds_respected(self):
        known = common.load_known_e5b()
        obs = common.load_observations("A_sink_clean", known)
        # A start above the bound is clipped into it; the optimizer never leaves the bounds.
        fit = baselines.fit_fd_inverse(obs, known, {"sink": True, "starts": [(1, 1, 1, 0.9)], "max_nfev": 3})
        f = fit["runs"][0]["sink_fraction"]
        self.assertTrue(baselines.SINK_BOUNDS[0] <= f <= baselines.SINK_BOUNDS[1] + 1e-12, f)


class E5bRoutingTests(unittest.TestCase):
    """E5B-DATA-ROUTING-01: sink data must not be scored against the base-model truth."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        patcher = mock.patch.object(common, "RESULTS_DIR", self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_base_runners_reject_sink_data_before_reading_or_output(self):
        runners = {"E0": lambda d: rx.run_e0(d), "E2": lambda d: rx.run_e2(d, 0),
                   "E3": lambda d: rx.run_e3(d, 0)}
        with mock.patch.object(common, "load_known", side_effect=AssertionError("unexpected data read")), \
                mock.patch.object(common, "load_observations", side_effect=AssertionError("unexpected data read")):
            for exp, run in runners.items():
                for name in ("A_sink_clean", "A_sink_noisy"):
                    with self.subTest(exp=exp, data=name), self.assertRaisesRegex(ValueError, "base-model truth"):
                        run(name)
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_cli_rejects_sink_data_for_base_experiments(self):
        for exp in ("E0", "E2", "E3"):
            err = io.StringIO()
            with self.subTest(exp=exp), redirect_stderr(err), \
                    mock.patch.object(rx, "_dispatch", side_effect=AssertionError("dispatched")), \
                    self.assertRaises(SystemExit) as stopped:
                rx.main(["--exp", exp, "--data", "A_sink_noisy"])
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn("base-model truth", err.getvalue())

    def test_base_data_still_accepted(self):
        for name in sorted(rx.BASE_DATA):
            rx._validate_base_data(name, "E3")  # no error

    def test_mismatch_data_rejected_by_base_runners(self):
        # BASE-MISMATCH-REFERENCE-01: A_mismatch's truth is mismatch_reference, which E0/E2/E3 never load.
        self.assertNotIn("A_mismatch", rx.BASE_DATA)
        runners = {"E0": lambda d: rx.run_e0(d), "E2": lambda d: rx.run_e2(d, 0),
                   "E3": lambda d: rx.run_e3(d, 0)}
        with mock.patch.object(common, "load_known", side_effect=AssertionError("unexpected data read")), \
                mock.patch.object(common, "load_observations", side_effect=AssertionError("unexpected data read")):
            for exp, run in runners.items():
                with self.subTest(exp=exp), self.assertRaisesRegex(ValueError, "A_mismatch only by E5"):
                    run("A_mismatch")
        for exp in runners:
            err = io.StringIO()
            with self.subTest(cli=exp), redirect_stderr(err), \
                    mock.patch.object(rx, "_dispatch", side_effect=AssertionError("dispatched")), \
                    self.assertRaises(SystemExit) as stopped:
                rx.main(["--exp", exp, "--data", "A_mismatch"])
            self.assertEqual(stopped.exception.code, 2)
        self.assertEqual(list(self.tmp.iterdir()), [])


class E5bFdExitTests(unittest.TestCase):
    """E5B-FD-EXIT-01: an unconverged FD fit must show in the summary status and the CLI exit code."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        patcher = mock.patch.object(common, "RESULTS_DIR", self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _fake_fit(fail_sink):
        # Coarse-grid stand-in for fit_fd_inverse so the runner's evaluation path runs quickly.
        def fit(obs, known, experiment=None):
            sink = bool((experiment or {}).get("sink"))
            if sink and fail_sink:
                return {"status": "failed", "reason": "no start converged", "runs": [], "wall_time_s": 0.0, "nx": 41}
            return {"status": "completed", "Bi": {"S": 1.0, "E": 0.4, "N": 0.25},
                    "sink_fraction": 0.1 if sink else None, "nx": 41, "sensor_rms_C": 0.0,
                    "jacobian": {}, "n_converged": 1, "selected_start": [1.0], "runs": [{}], "wall_time_s": 0.0}
        return fit

    def _run_cli(self, fail_sink):
        out = io.StringIO()
        with mock.patch.object(baselines, "fit_fd_inverse", side_effect=self._fake_fit(fail_sink)), \
                mock.patch("sys.stdout", out), self.assertRaises(SystemExit) as stopped:
            rx.main(["--exp", "E5b_fd"])
        summary = json.loads((self.tmp / "e5b" / "fd_A_sink" / "summary.json").read_text())
        return stopped.exception.code, summary, json.loads(out.getvalue())

    def test_partial_failure_exits_2_and_is_named(self):
        code, summary, printed = self._run_cli(fail_sink=True)
        self.assertEqual(code, 2)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(printed["status"], "failed")
        self.assertEqual(sorted(summary["failed_fits"]), ["A_sink_clean/sink", "A_sink_noisy/sink"])
        self.assertEqual(summary["results"]["A_sink_clean"]["nosink"]["status"], "completed")

    def test_all_converged_exits_0(self):
        code, summary, printed = self._run_cli(fail_sink=False)
        self.assertEqual(code, 0)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["failed_fits"], [])

    def test_no_successful_fields_preserves_failures_and_skips_only_empty_plots(self):
        for fail_all in (True, False):
            with self.subTest(fail_all=fail_all), tempfile.TemporaryDirectory() as tmp:
                successful = self._fake_fit(False)

                def fit(obs, known, experiment=None):
                    if fail_all or obs.name == "A_sink_clean":
                        return {"status": "failed", "reason": "injected nonconvergence",
                                "runs": [], "wall_time_s": 0.0, "nx": 41}
                    return successful(obs, known, experiment)

                with mock.patch.object(common, "RESULTS_DIR", Path(tmp)), \
                        mock.patch.object(baselines, "fit_fd_inverse", side_effect=fit), \
                        mock.patch.object(rx, "plot_compare") as plot, \
                        mock.patch("sys.stdout", io.StringIO()), \
                        self.assertRaises(SystemExit) as stopped:
                    rx.main(["--exp", "E5b_fd"])
                self.assertEqual(stopped.exception.code, 2)
                summary = json.loads((Path(tmp) / "e5b/fd_A_sink/summary.json").read_text())
                self.assertEqual(summary["status"], "failed")
                self.assertEqual(len(summary["failed_fits"]), 4 if fail_all else 2)
                import csv
                with (Path(tmp) / "metrics.csv").open() as file:
                    statuses = [row["status"] for row in csv.DictReader(file)]
                self.assertEqual(statuses.count("failed"), 4 if fail_all else 2)
                self.assertEqual(plot.call_count, 0 if fail_all else 1)
                if not fail_all:
                    self.assertEqual(len(plot.call_args.args[1]), 2)


class LocalIdentifiabilityTests(unittest.TestCase):
    """E5B-COVARIANCE-01 and E5B-RANK-PLOT-01."""

    def _J(self, singular_values, seed=0):
        rng = np.random.default_rng(seed)
        U, _ = np.linalg.qr(rng.normal(size=(10, 4)))
        V, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        return U @ np.diag(singular_values) @ V.T

    def test_ill_conditioned_full_rank_gives_finite_large_std(self):
        out = rx._local_identifiability(self._J([1, 0.1, 0.01, 1e-9]), (0.1,), 20.0, rx.E5B_PARAMS)
        self.assertEqual(out["rank"], 4)
        self.assertTrue(out["identifiable"])
        std = np.array(list(out["rel_std"]["0.1C"].values()))
        self.assertTrue(np.all(np.isfinite(std)))
        self.assertGreater(std.max(), 1e3)  # weak direction shows up as huge uncertainty, not as an error

    def test_rank_deficient_has_no_covariance_and_still_plots(self):
        J = self._J([1, 0, 0, 0])
        out = rx._local_identifiability(J, (0.1,), 20.0, rx.E5B_PARAMS)
        self.assertEqual(out["rank"], 1)
        self.assertFalse(out["identifiable"])
        self.assertNotIn("correlation", out)
        self.assertNotIn("rel_std", out)
        cases = {"base_model_no_sink": out, "sink_f_known": out, "sink_f_unknown": out}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ident.png"
            rx._plot_e5b_ident(cases, 0.1, np.ones(10), common.load_observations("A_sink_clean",
                               common.load_known_e5b()), path)
            self.assertTrue(path.exists())

    def test_svd_path_reproduces_saved_identifiability_run(self):
        # The saved run used the previous inv(J^T J) code; rerun the real pipeline into a
        # temporary results dir and compare every case's std and correlation.
        saved = json.loads((common.RESULTS_DIR / "e5b" / "ident" / "summary.json").read_text())
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(common, "RESULTS_DIR", Path(tmp)):
            fresh = common.to_jsonable(rx.run_e5b_ident())
        for name, rec in saved["cases"].items():
            new = fresh["cases"][name]
            self.assertEqual(new["rank"], rec["rank"], name)
            for sig, vals in rec["rel_std"].items():
                for p, v in vals.items():
                    self.assertAlmostEqual(new["rel_std"][sig][p], v, delta=1e-9, msg=f"{name} {sig} {p}")
            np.testing.assert_allclose(new["correlation"], rec["correlation"], atol=1e-9, rtol=0)


if __name__ == "__main__":
    unittest.main()
