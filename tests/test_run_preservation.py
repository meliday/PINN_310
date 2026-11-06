"""E2-RUN-PRESERVE-01: an existing run must not be replaced unless --overwrite is given."""

import io
from pathlib import Path
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest import mock

from classroom_heat_pinn_data.src import common, evaluate, pinn, run_experiments as rx


def _blocked(*args, **kwargs):
    raise AssertionError("must not be reached: the run should be refused first")


class RunPreservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        patcher = mock.patch.object(common, "RESULTS_DIR", self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _existing(self, *parts):
        d = self.tmp.joinpath(*parts)
        d.mkdir(parents=True)
        ckpt = d / "checkpoint.pt"
        ckpt.write_bytes(b"previous successful run")
        return ckpt

    def test_e2_rerun_refused_before_training(self):
        ckpt = self._existing("e2", "A_clean_seed0")
        with mock.patch.object(pinn, "train", _blocked), \
             mock.patch.object(evaluate, "load_reference", _blocked):
            with self.assertRaises(rx.RunExistsError):
                rx.run_e2("A_clean", 0, "base")
        self.assertEqual(ckpt.read_bytes(), b"previous successful run")
        self.assertFalse((self.tmp / "metrics.csv").exists())

    def test_failing_rerun_cannot_replace_previous_checkpoint(self):
        # Codex reproduction: a rerun whose last update is NaN used to overwrite the checkpoint.
        ckpt = self._existing("e2", "A_clean_C1_seed0")

        def nan_train(model, known, exp, obs=None):
            return [], {"status": "failed", "stage": "adam", "reason": "non-finite"}

        with mock.patch.object(pinn, "train", nan_train):
            with self.assertRaises(rx.RunExistsError):
                rx.run_e2("A_clean", 0, "C1")
        self.assertEqual(ckpt.read_bytes(), b"previous successful run")

    def test_e1_and_e0_reruns_refused_before_truth_access(self):
        self._existing("e1", "forward_seed0")
        self._existing("e0", "A_clean")
        with mock.patch.object(evaluate, "load_truth_bi", _blocked), \
             mock.patch.object(rx, "tool_checks", _blocked):
            with self.assertRaises(rx.RunExistsError):
                rx.run_e1(0)
            with self.assertRaises(rx.RunExistsError):
                rx.run_e0("A_clean")

    def test_cli_exits_3_with_message(self):
        self._existing("e2", "A_clean_seed0")
        err = io.StringIO()
        with mock.patch.object(pinn, "train", _blocked), redirect_stderr(err):
            with self.assertRaises(SystemExit) as cm:
                rx.main(["--exp", "E2", "--data", "A_clean", "--seed", "0"])
        self.assertEqual(cm.exception.code, 3)
        self.assertIn("--overwrite", err.getvalue())

    def test_overwrite_replaces_existing_run(self):
        ckpt = self._existing("e2", "A_clean_seed0")
        tiny = {"adam_steps": 2, "lbfgs_iters": 0, "n_interior": 40, "n_wall": 8,
                "hidden": 8, "depth": 1}
        with mock.patch.dict(rx.E2_EXPERIMENT, tiny):
            s = rx.run_e2("A_clean", 0, "base", overwrite=True)
        self.assertEqual(s["status"], "completed")
        self.assertNotEqual(ckpt.read_bytes(), b"previous successful run")
        self.assertTrue((self.tmp / "metrics.csv").exists())

    def test_empty_directory_is_not_treated_as_existing_run(self):
        (self.tmp / "e2" / "A_clean_seed0").mkdir(parents=True)
        rx._claim_run_dir(self.tmp / "e2" / "A_clean_seed0")  # no error


if __name__ == "__main__":
    unittest.main()
