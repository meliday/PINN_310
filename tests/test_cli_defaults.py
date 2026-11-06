"""E3-CLI-DATA-01: per-experiment default data and noise metadata that match the data."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from classroom_heat_pinn_data.src import common, run_experiments as rx


class _Stop(Exception):
    pass


def _capture(name):
    """Spy for a run_* function: records its arguments and stops before any work."""
    seen = {}

    def spy(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        raise _Stop

    return seen, mock.patch.object(rx, name, spy)


class CliDefaultDataTests(unittest.TestCase):
    def _data_for(self, argv, runner):
        seen, patch = _capture(runner)
        with patch, self.assertRaises(_Stop):
            rx.main(argv)
        return seen["args"][0]

    def test_e3_defaults_to_noisy(self):
        self.assertEqual(self._data_for(["--exp", "E3"], "run_e3"), "A_noisy")

    def test_e0_and_e2_default_to_clean(self):
        self.assertEqual(self._data_for(["--exp", "E0"], "run_e0"), "A_clean")
        self.assertEqual(self._data_for(["--exp", "E2"], "run_e2"), "A_clean")

    def test_explicit_data_is_respected(self):
        self.assertEqual(self._data_for(["--exp", "E3", "--data", "B_noisy"], "run_e3"), "B_noisy")


class NoiseMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        patcher = mock.patch.object(common, "RESULTS_DIR", self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_sigma_follows_data_name(self):
        self.assertEqual(rx.noise_sigma_C("A_noisy"), 0.1)
        self.assertEqual(rx.noise_sigma_C("B_noisy"), 0.1)
        self.assertEqual(rx.noise_sigma_C("A_clean"), 0.0)
        with self.assertRaises(ValueError):
            rx.noise_sigma_C("A_mismatch")

    def test_unsupported_data_fails_before_any_output(self):
        with self.assertRaises(ValueError):
            rx.run_e3("A_mismatch", 0)
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_clean_run_records_no_noise_and_skips_uncertainty(self):
        tiny = {"adam_steps": 2, "lbfgs_iters": 0, "n_interior": 40, "n_wall": 8, "hidden": 8, "depth": 1}
        with mock.patch.dict(rx.E2_CONFIGS["C1"], tiny):
            s = rx.run_e3("A_clean", 0)
        saved = json.loads((self.tmp / "e3" / "A_clean_seed0" / "summary.json").read_text())
        self.assertEqual(saved["noise_sigma_C"], 0.0)
        self.assertIn("skipped", s["fd_inverse"]["local_uncertainty"])
        self.assertNotIn("rel_std_bi", s["fd_inverse"]["local_uncertainty"])


if __name__ == "__main__":
    unittest.main()
