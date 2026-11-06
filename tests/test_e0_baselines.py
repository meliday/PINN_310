"""E0 checks: truth isolation of the fitting path, units, orientation, flux signs."""

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from classroom_heat_pinn_data.src import baselines, common, evaluate, fd, run_experiments

# Records every file opened while armed; the hook cannot be removed, so it is gated.
_OPENED = []
_ARMED = [False]


def _audit(event, args):
    if _ARMED[0] and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        _OPENED.append(os.path.realpath(os.fsdecode(args[0])))


sys.addaudithook(_audit)
TRUTH_REAL = {os.path.realpath(common.DATA_DIR / f) for f in common.TRUTH_FILES}


class _Watch:
    def __enter__(self):
        _OPENED.clear()
        _ARMED[0] = True
        return _OPENED

    def __exit__(self, *exc):
        _ARMED[0] = False


def _blocked(*args, **kwargs):
    raise AssertionError("truth loader called from a fitting path")


class TruthIsolationTests(unittest.TestCase):
    def test_fitting_path_opens_no_truth_file(self):
        with mock.patch.object(evaluate, "load_truth_config", _blocked), \
             mock.patch.object(evaluate, "load_reference", _blocked), \
             mock.patch.object(evaluate, "load_truth_bi", _blocked), _Watch() as opened:
            known = common.load_known()
            obs = common.load_observations("A_clean", known)
            q = np.array([[0.5, 0.4], [0.1, 0.1]])
            baselines.reconstruct_rbf(obs.xy, obs.theta, q)
            fit = baselines.fit_fd_inverse(obs, known, {"starts": [(1, 1, 1)]})
        self.assertEqual(fit["status"], "completed")
        self.assertTrue(any(p.endswith("sensors_A_clean.csv") for p in opened))
        self.assertFalse(set(opened) & TRUTH_REAL, f"truth opened: {set(opened) & TRUTH_REAL}")

    def test_watch_detects_truth_access_through_alias(self):
        # Positive control: the hook must see a truth read, including via a symlink.
        with tempfile.TemporaryDirectory() as tmp:
            alias = Path(tmp) / "alias.json"
            alias.symlink_to(common.DATA_DIR / "config.json")
            with _Watch() as opened:
                alias.read_text()
            self.assertTrue(set(opened) & TRUTH_REAL)

    def test_known_has_no_bi(self):
        known = common.load_known()
        self.assertFalse(any(k.lower().startswith("bi") for k in known))


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        shutil.copy(common.DATA_DIR / "known.json", self.tmp)
        self.known = common.load_known(self.tmp)

    def test_rejects_truth_or_unknown_file_names(self):
        with self.assertRaises(ValueError):
            common.load_observations("../config", self.known, self.tmp)

    def test_unit_tolerance_on_rounded_csv(self):
        src = (common.DATA_DIR / "sensors_A_clean.csv").read_text().splitlines()
        (self.tmp / "sensors_A_clean.csv").write_text("\n".join(src) + "\n")
        obs = common.load_observations("A_clean", self.known, self.tmp)
        self.assertEqual(obs.xy.shape, (10, 2))
        cols = src[1].split(",")
        cols[4] = f"{float(cols[4]) + 0.01:.4f}"  # 0.01 C mismatch must be rejected
        (self.tmp / "sensors_A_clean.csv").write_text("\n".join([src[0], ",".join(cols)] + src[2:]) + "\n")
        with self.assertRaises(ValueError):
            common.load_observations("A_clean", self.known, self.tmp)

    def test_missing_values_rejected(self):
        src = (common.DATA_DIR / "sensors_A_clean.csv").read_text().splitlines()
        cols = src[1].split(",")
        cols[3] = cols[4] = ""  # NaN would otherwise pass the unit tolerance check
        (self.tmp / "sensors_A_clean.csv").write_text("\n".join([src[0], ",".join(cols)] + src[2:]) + "\n")
        with self.assertRaises(ValueError):
            common.load_observations("A_clean", self.known, self.tmp)


class RunBookkeepingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_unconverged_fit_is_failed(self):
        known = common.load_known()
        obs = common.load_observations("A_clean", known)
        fit = baselines.fit_fd_inverse(obs, known, {"starts": [(1, 1, 1)], "max_nfev": 1})
        self.assertEqual(fit["status"], "failed")
        self.assertEqual(fit["runs"][0]["status"], 0)
        self.assertFalse(fit["runs"][0]["ok"])

    def test_metrics_upsert_keeps_other_datasets(self):
        path = self.tmp / "metrics.csv"
        row = lambda data, rmse: evaluate.metrics_row(
            exp_id="E0", method="RBF_tps", data_file=data, status="completed", q1_rmse_C=rmse)
        run_experiments.write_metrics([row("sensors_A_clean.csv", 1.0)], path)
        run_experiments.write_metrics([row("sensors_B_clean.csv", 2.0)], path)
        run_experiments.write_metrics([row("sensors_A_clean.csv", 3.0)], path)
        with open(path, newline="") as fh:
            got = {r["data_file"]: r["q1_rmse_C"] for r in run_experiments.csv.DictReader(fh)}
        self.assertEqual(got, {"sensors_A_clean.csv": "3.0", "sensors_B_clean.csv": "2.0"})

    def test_failed_tool_check_stops_before_fitting(self):
        def fake_checks(known, obs):
            return {"passed": {"mms_order_gt_1.8": True, "energy_lt_1e-8": False}}

        with mock.patch.object(common, "RESULTS_DIR", self.tmp), \
             mock.patch.object(run_experiments, "tool_checks", fake_checks), \
             mock.patch.object(baselines, "fit_fd_inverse", _blocked), \
             mock.patch.object(baselines, "reconstruct_rbf", _blocked):
            s = run_experiments.run_e0("A_clean")
        self.assertEqual(s["status"], "failed")
        self.assertIn("energy_lt_1e-8", s["reason"])
        self.assertTrue((self.tmp / "e0" / "A_clean" / "summary.json").exists())
        with open(self.tmp / "metrics.csv", newline="") as fh:
            statuses = {r["method"]: r["status"] for r in run_experiments.csv.DictReader(fh)}
        self.assertEqual(statuses, {"RBF_tps": "failed", "FD_inverse": "failed"})


class OrientationAndFluxTests(unittest.TestCase):
    def test_sampling_uses_ij_order(self):
        x, y = np.linspace(0, 1, 101), np.linspace(0, 0.8, 81)
        X, Y = np.meshgrid(x, y, indexing="ij")
        field = X + 3 * Y  # asymmetric linear field: bilinear is exact
        pts = np.array([[0.3, 0.04], [0.95, 0.55], [0.123, 0.777]])
        np.testing.assert_allclose(fd.sample(x, y, field, pts), pts[:, 0] + 3 * pts[:, 1], atol=1e-12)
        ix, iy = fd.node_indices(x, y, pts[:2])
        np.testing.assert_allclose(field[ix, iy], pts[:2, 0] + 3 * pts[:2, 1], atol=1e-12)
        self.assertIsNone(fd.node_indices(x, y, pts))

    def test_signed_shares_keep_inflow(self):
        info = {"q_in": 1.0, "total_flux": {"S": 1.3, "E": 0.0, "N": -0.3, "W": 0.0}}
        fl = fd.heat_flows(info)
        self.assertAlmostEqual(fl["share_signed"]["N"], -0.3)
        self.assertAlmostEqual(fl["share_signed"]["S"], 1.3)  # >100 % allowed, not clipped
        self.assertAlmostEqual(fl["share_outflow"]["S"], 1.0)
        self.assertAlmostEqual(fl["inflow"]["N"], -0.3)
        self.assertLess(fl["energy_balance_relerr"], 1e-12)

    def test_wrapper_energy_balance_and_flux_signs(self):
        known = common.load_known()
        _, _, _, info = fd.solve_field((1.0, 1.0, 1.0), known)
        fl = fd.heat_flows(info)
        self.assertLess(fl["energy_balance_relerr"], 1e-8)
        self.assertGreater(fl["flux"]["S"], 0)  # cold outdoor side loses heat


class FdInverseRecoveryTests(unittest.TestCase):
    def test_clean_recovery_within_half_percent(self):
        known = common.load_known()
        obs = common.load_observations("A_clean", known)
        fit = baselines.fit_fd_inverse(obs, known, {"starts": [(5, 5, 5)]})
        # Truth used only to score the finished fit.
        bi_true = json.loads((common.DATA_DIR / "config.json").read_text())["Bi_true"]
        for w in common.ROBIN_WALLS:
            self.assertLess(abs(fit["Bi"][w] / bi_true[w] - 1), 5e-3, w)
        self.assertEqual(fit["jacobian"]["rank"], 3)


if __name__ == "__main__":
    unittest.main()
