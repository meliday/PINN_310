"""X1 data access: known_x1 + sensors load without touching X1 truth (LOG-079 next step 7)."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

from classroom_heat_pinn_data.src import common, run_experiments as rx
from classroom_heat_pinn_data.synthetic import generate_x1, solver_wall

_OPENED = []
_ARMED = [False]


def _audit(event, args):
    if _ARMED[0] and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        _OPENED.append(os.path.realpath(os.fsdecode(args[0])))


sys.addaudithook(_audit)
X1_TRUTH_REAL = {os.path.realpath(common.X1_DIR / f) for f in common.X1_TRUTH_FILES}


class _Watch:
    def __enter__(self):
        _OPENED.clear()
        _ARMED[0] = True
        return _OPENED

    def __exit__(self, *exc):
        _ARMED[0] = False


class X1LoaderTests(unittest.TestCase):
    def test_truth_and_training_files_cover_the_bundle(self):
        on_disk = {p.name for p in common.X1_DIR.iterdir()} - {"verification_x1.json"}
        self.assertEqual(on_disk, common.X1_TRAINING_FILES | common.X1_TRUTH_FILES)
        self.assertFalse(common.X1_TRAINING_FILES & common.X1_TRUTH_FILES)

    def test_known_gives_wall_geometry_matching_the_generator(self):
        known = common.load_known_x1()
        self.assertEqual(common.solids(known), [generate_x1.WALL])
        self.assertEqual(common.solids(common.load_known()), [])
        # the solver accepts the loaded geometry as is
        solver_wall.check_aligned(*solver_wall.solver.make_grid(101)[:2], common.solids(known))

    def test_observations_load_from_x1_dir_without_truth(self):
        with _Watch() as opened:
            known = common.load_known_x1()
            clean = common.load_observations("A_x1_clean", known)
            noisy = common.load_observations("A_x1_noisy", known)
        self.assertTrue(any(p.endswith("x1/sensors_A_x1_clean.csv") for p in opened))
        self.assertFalse(set(opened) & X1_TRUTH_REAL, f"truth opened: {set(opened) & X1_TRUTH_REAL}")
        base = common.load_observations("A_clean", common.load_known())
        np.testing.assert_array_equal(clean.xy, base.xy)  # same layout A
        self.assertGreater(np.abs(noisy.theta - clean.theta).max(), 0)
        self.assertFalse(common.inside_solid(clean.xy, known).any())

    def test_watch_detects_x1_truth_access(self):
        # Positive control for the hook, through a symlink alias.
        with tempfile.TemporaryDirectory() as tmp:
            alias = Path(tmp) / "alias.json"
            alias.symlink_to(common.X1_DIR / "config_x1.json")
            with _Watch() as opened:
                alias.read_text()
        self.assertTrue(set(opened) & X1_TRUTH_REAL)

    def test_truth_names_are_rejected(self):
        known = common.load_known_x1()
        for name in ("../x1/config_x1", "x1", "A_x1", "reference_x1"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                common.load_observations(name, known)

    def test_known_with_bi_or_without_solids_is_refused(self):
        base = json.loads((common.X1_DIR / "known_x1.json").read_text())
        for change in ({"Bi": {"S": 1.0}}, {"solids": []}, {"solids": [{"x0": 0.6, "x1": 0.0, "y0": 0.3, "y1": 0.34}]}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "known_x1.json").write_text(json.dumps({**base, **change}))
                with self.assertRaises(ValueError):
                    common.load_known_x1(tmp)

    def test_sensor_inside_wall_is_refused(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        rows = (common.X1_DIR / "sensors_A_x1_clean.csv").read_text().splitlines()
        rows[1] = "0,0.30,0.32,0.5,10.0"  # inside the wall
        (tmp / "sensors_A_x1_clean.csv").write_text("\n".join(rows) + "\n")
        known = common.load_known_x1()
        with self.assertRaisesRegex(ValueError, "inside a solid"):
            common.load_observations("A_x1_clean", known, tmp)
        # on the surface is fluid
        rows[1] = "0,0.30,0.34,0.5,10.0"
        (tmp / "sensors_A_x1_clean.csv").write_text("\n".join(rows) + "\n")
        common.load_observations("A_x1_clean", known, tmp)

    def test_base_runners_reject_x1_data(self):
        for exp in ("E0", "E2", "E3"):
            with self.subTest(exp=exp), self.assertRaisesRegex(ValueError, "base-model truth"):
                rx._validate_base_data("A_x1_clean", exp)


if __name__ == "__main__":
    unittest.main()
