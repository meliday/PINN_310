"""§18 stage-1 FD repeat baseline: inputs, truth isolation, aggregation (LOG-117)."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from classroom_heat_pinn_data.src import common, stage1_fd as s1

_OPENED, _ARMED = [], [False]


def _audit(event, args):
    if _ARMED[0] and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        _OPENED.append(os.path.realpath(os.fsdecode(args[0])))


sys.addaudithook(_audit)
TRUTH = ({os.path.realpath(s1.STAGE1_DIR / f) for f in s1.STAGE1_TRUTH_FILES}
         | {os.path.realpath(d / f) for d, fs in ((common.DATA_DIR, common.TRUTH_FILES), (common.E5B_DIR, common.E5B_TRUTH_FILES),
                                                  (common.X1_DIR, common.X1_TRUTH_FILES), (common.X2_DIR, common.X2_TRUTH_FILES))
            for f in fs})


class Stage1InputTests(unittest.TestCase):
    def test_noisy_observation_is_clean_plus_scaled_standard_normal(self):
        known, clean = s1.load_observation("wall_sink_eq", "C")
        _, noisy = s1.load_observation("wall_sink_eq", "C", 0.2, 3)
        with np.load(s1.STAGE1_DIR / "noise_standard_normal.npz") as f:
            z = f["z"][3]
        np.testing.assert_allclose(noisy.theta, clean.theta + 0.2 / 20.0 * z, rtol=0, atol=1e-15)
        csv = pd.read_csv(s1.STAGE1_DIR / "sensors_wall_sink_eq_C_clean.csv")
        np.testing.assert_array_equal(clean.theta, csv["theta_obs"].to_numpy())
        self.assertFalse(common.inside_solid(clean.xy, known).any())
        self.assertGreater(known["heater"]["A"], common.load_known()["heater"]["A"])  # equal-heat scaling

    def test_unknown_inputs_refused(self):
        for cond, layout in (("x3", "A"), ("base", "B")):
            with self.subTest(cond=cond, layout=layout), self.assertRaises(ValueError):
                s1.load_observation(cond, layout)

    def test_preregistered_task_count(self):
        t = s1.tasks()
        self.assertEqual(len(t), 6 * 2 * (1 + 3 * 10) + 8)
        self.assertEqual(len(set(t)), len(t))
        c = s1.tasks("confirm")  # LOG-123: noisy only, 30 new draws
        self.assertEqual(len(c), 6 * 2 * 3 * 30)
        self.assertTrue(all(t[2] > 0 and t[5] == "confirm" for t in c))

    def test_confirmation_noise_is_independent_of_pilot(self):
        with np.load(s1.STAGE1_DIR / "noise_confirm_standard_normal.npz") as f:
            zc, seeds = f["z"], f["seeds"]
        with np.load(s1.STAGE1_DIR / "noise_standard_normal.npz") as f:
            zp, pseeds = f["z"], f["seeds"]
        self.assertEqual(zc.shape, (30, 10))
        self.assertFalse(set(seeds.tolist()) & set(pseeds.tolist()))
        np.testing.assert_array_equal(zc[4], np.random.default_rng(int(seeds[4])).standard_normal(10))
        _, clean = s1.load_observation("sink", "A")
        _, noisy = s1.load_observation("sink", "A", 0.1, 4, noise="confirm")
        np.testing.assert_allclose(noisy.theta, clean.theta + 0.1 / 20.0 * zc[4], rtol=0, atol=1e-15)

    def test_fitting_reads_no_truth(self):
        fake = {"status": "completed", "Bi": {"S": 1.0, "E": 0.4, "N": 0.25}, "sink_fraction": 0.1,
                "sensor_rms_C": 0.0, "n_converged": 1}
        with mock.patch.object(s1.baselines, "fit_fd_inverse", side_effect=lambda o, k, e: {**fake, "sink_fraction": 0.1 if e["sink"] else None}), \
                mock.patch.object(s1.xf, "sensor_jacobian", return_value=np.eye(10, 4)):
            _OPENED.clear(); _ARMED[0] = True
            try:
                recs = [s1.fit_task(t) for t in s1.tasks()[::7]]
            finally:
                _ARMED[0] = False
        self.assertFalse(set(_OPENED) & TRUTH, set(_OPENED) & TRUTH)
        self.assertTrue(all(r["status"] == "completed" for r in recs))


class Stage1AggregateTests(unittest.TestCase):
    def test_bias_rmse_and_coverage(self):
        rows = []
        for e, s in ((0.1, 0.1), (-0.1, 0.1), (0.3, 0.1)):  # |log err| <= 1.96 s for the first two only
            r = {"cond": "base", "layout": "A", "sigma_C": 0.1, "nx": 101, "status": "completed", "has_sink": False,
                 "at_bound": False, "fit_time_s": 1.0, "q1_rmse_C": 0.05}
            for w in common.ROBIN_WALLS:
                r.update({f"Bi_{w}_relerr": np.exp(e) - 1, f"Bi_{w}_logerr": e, f"Bi_{w}_local_std": s})
            rows.append(r)
        rows.append({**rows[0], "status": "failed"})
        g = s1.aggregate(rows)[0]
        self.assertEqual((g["n"], g["n_failed"]), (4, 1))
        self.assertAlmostEqual(g["Bi_S_coverage_1.96"], 2 / 3)
        rel = np.exp([0.1, -0.1, 0.3]) - 1
        self.assertAlmostEqual(g["Bi_S_bias"], rel.mean())
        self.assertAlmostEqual(g["Bi_S_rmse"], np.sqrt(np.mean(rel ** 2)))


class Stage1RunTests(unittest.TestCase):
    def test_small_real_run_and_refusal(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        log = tmp / "logs"
        with mock.patch.object(s1, "tasks", return_value=[("base", "A", 0.0, None, 101, "pilot"), ("wall_eq", "C", 0.1, 2, 101, "pilot")]), \
                mock.patch.object(common, "RESULTS_DIR", tmp):
            summary = s1.run(tmp / "run", workers=1)
        self.assertEqual((summary["n_fits"], summary["n_failed"]), (2, 0))
        base = next(g for g in summary["groups"] if g["cond"] == "base")
        self.assertLess(abs(base["Bi_S_bias"]), 1e-3)  # clean base on the 101 grid (E0-like fit)
        self.assertIn("commit", summary["provenance"])
        # LOG-121 STAGE1-EVAL-PROVENANCE-01: every evaluation file is hashed
        self.assertEqual(len(summary["evaluation_sha256"]), 7)
        self.assertTrue(any(k.endswith("config_stage1.json") for k in summary["evaluation_sha256"]))
        self.assertGreater(summary["common_holdout_nodes"], 31000)
        self.assertTrue(all(g["q1_common_rmse_C_mean"] is not None for g in summary["groups"]))
        self.assertTrue((tmp / "run" / "fits.csv").exists() and (tmp / "run" / "groups.csv").exists())
        self.assertIn("config_stage1.json", (log / "truth_access.log").read_text())
        with self.assertRaises(FileExistsError):
            s1.run(tmp / "run", workers=1)


if __name__ == "__main__":
    unittest.main()
