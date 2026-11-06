"""FD with internal walls (+ sink) and the X1-d / X2 FD runners (LOG-088)."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from classroom_heat_pinn_data.src import common, evaluate, extensions_fd as xf, fd, run_experiments as rx
from classroom_heat_pinn_data.synthetic import solver, solver_wall

BI = {"S": 1.0, "E": 0.4, "N": 0.25}  # fixed test values, not read from truth files

_OPENED, _ARMED = [], [False]


def _audit(event, args):
    if _ARMED[0] and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        _OPENED.append(os.path.realpath(os.fsdecode(args[0])))


sys.addaudithook(_audit)
ALL_TRUTH = {os.path.realpath(d / f) for d, files in ((common.DATA_DIR, common.TRUTH_FILES),
                                                      (common.E5B_DIR, common.E5B_TRUTH_FILES),
                                                      (common.X1_DIR, common.X1_TRUTH_FILES),
                                                      (common.X2_DIR, common.X2_TRUTH_FILES)) for f in files}


class FdWallTests(unittest.TestCase):
    def test_open_room_paths_unchanged(self):
        known = common.load_known()
        _, _, a, _ = fd.solve_field(BI, known, nx=101)
        _, _, b, _ = solver.solve(101, BI, common.theta_inf(known), heaters=common.heaters(known))
        np.testing.assert_array_equal(a, b)

    def test_wall_field_matches_wall_solver(self):
        known = common.load_known_x1()
        _, _, a, info = fd.solve_field(BI, known, nx=101)
        _, _, b, _ = solver_wall.solve(101, BI, common.theta_inf(known), solids=common.solids(known),
                                       heaters=common.heaters(known))
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isnan(a).any())

    def test_sink_removes_fraction_of_fluid_heater_power(self):
        known = common.load_known_x2()
        _, _, _, info = fd.solve_field(BI, known, nx=101, sink_fraction=0.1)
        self.assertAlmostEqual(info["sink_power"] / info["heater_power"], 0.1, places=12)
        x, y, _ = solver.make_grid(101)
        X, Y = np.meshgrid(x, y, indexing="ij")
        vol = solver_wall.fluid_volumes(x, y, common.solids(known))
        H = float((solver.heater_source(X, Y, common.heaters(known)) * vol).sum())
        self.assertAlmostEqual(info["heater_power"], H, places=12)
        flows = fd.heat_flows(info)
        self.assertLess(flows["energy_balance_relerr"], 1e-10)

    def test_known_x2_and_files(self):
        known = common.load_known_x2()
        self.assertTrue(common.solids(known) and common.sink_geometry(known))
        on_disk = {p.name for p in common.X2_DIR.iterdir()} - {"verification_x2.json"}
        self.assertEqual(on_disk, common.X2_TRAINING_FILES | common.X2_TRUTH_FILES)
        with tempfile.TemporaryDirectory() as tmp:
            k = json.loads((common.X2_DIR / "known_x2.json").read_text())
            k["sink"]["strength"] = 0.1
            (Path(tmp) / "known_x2.json").write_text(json.dumps(k))
            with self.assertRaises(ValueError):
                common.load_known_x2(tmp)

    def test_q1_excludes_solid_nodes_and_reports_tip(self):
        known = common.load_known_x1()
        obs = common.load_observations("A_x1_clean", known)
        x, y, th, _ = fd.solve_field(BI, known, nx=201)
        ref = {"x": x, "y": y, "theta": th, "fluid": ~np.isnan(th),
               "wall_tip": np.zeros(th.shape, bool) | (np.hypot(*np.meshgrid(x - 0.6, y - 0.32, indexing="ij")) < 0.1)}
        q = evaluate.q1_metrics(th, ref, obs, known)
        self.assertEqual(q["q1_rmse_C"], 0.0)
        self.assertEqual(q["n_eval_points"], int((~np.isnan(th)).sum()) - 10)
        self.assertIn("q1_rmse_wall_tip_C", q)


class ExtensionRunnerTests(unittest.TestCase):
    def test_variants_per_condition(self):
        self.assertEqual([v[0] for v in xf.model_variants("base")], ["correct"])
        self.assertEqual([v[0] for v in xf.model_variants("wall")], ["correct", "ignore_all"])
        self.assertEqual([v[0] for v in xf.model_variants("wall_sink")],
                         ["correct", "ignore_all", "ignore_wall", "ignore_sink"])
        k = xf.variant_known(common.load_known_x2(), False, True)
        self.assertNotIn("solids", k)
        self.assertIn("sink", k)

    def test_fitting_reads_no_truth(self):
        fake = {"status": "completed", "Bi": BI, "sink_fraction": 0.1, "n_converged": 1, "sensor_rms_C": 0.0}

        def fit(obs, known, experiment=None):
            return {**fake, "sink_fraction": 0.1 if experiment.get("sink") else None}

        with mock.patch.object(xf.baselines, "fit_fd_inverse", side_effect=fit), \
                mock.patch.object(xf, "sensor_jacobian", return_value=np.eye(10, 4)):
            for cond in xf.CONDITIONS:
                _OPENED.clear(); _ARMED[0] = True
                try:
                    _, fits = xf.fit_condition(cond, 101)
                finally:
                    _ARMED[0] = False
                with self.subTest(cond=cond):
                    self.assertFalse(set(_OPENED) & ALL_TRUTH, set(_OPENED) & ALL_TRUTH)
                    self.assertEqual(len(fits), 2 * len(xf.model_variants(cond)))

    def test_existing_run_refused_before_fitting(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        for exp, sub in (("X1_fd", "x1/fd_A_x1"), ("X2_fd", "x2/fd_four_conditions")):
            (tmp / sub).mkdir(parents=True)
            (tmp / sub / "summary.json").write_text("{}")
            with self.subTest(exp=exp), mock.patch.object(common, "RESULTS_DIR", tmp), \
                    mock.patch.object(xf, "fit_condition", side_effect=AssertionError("fitted")), \
                    mock.patch("sys.stderr"), self.assertRaises(SystemExit) as stopped:
                rx.main(["--exp", exp])
            self.assertEqual(stopped.exception.code, 3)


if __name__ == "__main__":
    unittest.main()
