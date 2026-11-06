"""Per-experiment data defaults and E3/E5 evaluation-data contracts."""

from contextlib import redirect_stderr
import io
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

    def test_e5_defaults_to_mismatch(self):
        self.assertEqual(self._data_for(["--exp", "E5"], "run_e5"), "A_mismatch")

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


class E5DataContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        patcher = mock.patch.object(common, "RESULTS_DIR", self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_wrong_data_is_rejected_before_reading_or_output(self):
        with mock.patch.object(common, "load_known", side_effect=AssertionError("unexpected data read")) as load, \
                mock.patch.object(rx, "_claim_run_dir") as claim:
            for name in ("A_clean", "A_noisy", "B_clean", "B_noisy", "unknown", None):
                with self.subTest(data=name), self.assertRaisesRegex(ValueError, "E5 requires 'A_mismatch'"):
                    rx.run_e5(name)
            load.assert_not_called()
            claim.assert_not_called()
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_wrong_cli_data_exits_2_before_dispatch(self):
        with mock.patch.object(rx, "run_e5", side_effect=AssertionError("unexpected E5 run")) as runner:
            for name in ("A_clean", "A_noisy", "B_clean", "B_noisy", "unknown"):
                stderr = io.StringIO()
                with self.subTest(data=name), redirect_stderr(stderr), self.assertRaises(SystemExit) as stopped:
                    rx.main(["--exp", "E5", "--data", name])
                self.assertEqual(stopped.exception.code, 2)
                self.assertIn("E5 requires 'A_mismatch'", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())
            runner.assert_not_called()
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_wrong_data_cannot_overwrite_existing_output(self):
        run_dir = self.tmp / "e5" / "A_clean_seed0"
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "checkpoint.pt"
        checkpoint.write_bytes(b"previous checkpoint")
        with mock.patch.object(common, "load_known", side_effect=AssertionError("unexpected data read")), \
                self.assertRaisesRegex(ValueError, "E5 requires 'A_mismatch'"):
            rx.run_e5("A_clean", overwrite=True)
        self.assertEqual(checkpoint.read_bytes(), b"previous checkpoint")
        self.assertEqual([p for p in self.tmp.rglob("*") if p.is_file()], [checkpoint])

    def test_mismatch_data_runs_and_keeps_matching_evaluation(self):
        tiny = {"adam_steps": 2, "lbfgs_iters": 0, "n_interior": 40, "n_wall": 8, "hidden": 8, "depth": 1}
        with mock.patch.dict(rx.E2_CONFIGS["C1"], tiny):
            summary = rx.run_e5("A_mismatch", 0)
        saved = json.loads((self.tmp / "e5" / "A_mismatch_seed0" / "summary.json").read_text())
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(saved["data_file"], "sensors_A_mismatch.csv")
        self.assertLess(saved["truth"]["field_maxdiff_vs_reference_C"], 1e-8)
        self.assertAlmostEqual(saved["truth"]["shares"]["draft"], 0.1)
        log = (self.tmp / "logs" / "truth_access.log").read_text()
        self.assertIn("mismatch_reference.npz", log)
        self.assertNotIn("\treference.npz\t", log)


if __name__ == "__main__":
    unittest.main()
