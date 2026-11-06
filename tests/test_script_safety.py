"""Check import safety, working-directory independence and data preservation."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from classroom_heat_pinn_data.synthetic import generate_data, plot_overview


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "classroom_heat_pinn_data/synthetic/data"


class ScriptSafetyTests(unittest.TestCase):
    def run_python(self, args, cwd, **extra_env):
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1",
                   MPLCONFIGDIR=str(Path(cwd) / "mpl"), **extra_env)
        result = subprocess.run([sys.executable, "-B", *map(str, args)], cwd=cwd,
                                env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_imports_do_not_read_data_solve_or_write(self):
        code = '''
import importlib
from unittest import mock
import numpy, pandas, scipy.io
from classroom_heat_pinn_data.synthetic import solver
with mock.patch.object(solver, "solve", side_effect=AssertionError("solve on import")), \\
     mock.patch("builtins.open", side_effect=AssertionError("data open on import")), \\
     mock.patch("pathlib.Path.read_text", side_effect=AssertionError("data read on import")), \\
     mock.patch("pathlib.Path.mkdir", side_effect=AssertionError("mkdir on import")), \\
     mock.patch("numpy.load", side_effect=AssertionError("np.load on import")), \\
     mock.patch("numpy.savez", side_effect=AssertionError("np.savez on import")):
    for name in ("synthetic.generate_data", "synthetic.audit", "synthetic.plot_overview", "real.prepare_real"):
        importlib.import_module("classroom_heat_pinn_data." + name)
'''
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_python(["-c", code], tmp)
            self.assertEqual(result.stdout, "")
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_generation_refuses_existing_data_before_solving(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = Path(tmp) / "config.json"
            sentinel.write_text("original")
            with mock.patch.object(generate_data, "solve", side_effect=AssertionError("solve")):
                with self.assertRaises(FileExistsError):
                    generate_data.generate(tmp)
            self.assertEqual(sentinel.read_text(), "original")
            self.assertEqual(list(Path(tmp).iterdir()), [sentinel])

    def test_generation_cli_from_another_directory_preserves_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "generated"
            self.run_python([ROOT / "classroom_heat_pinn_data/synthetic/generate_data.py",
                             "--output-dir", output], tmp)
            for filename in ("reference.npz", "whatif_reference.npz", "mismatch_reference.npz"):
                with np.load(output / filename) as actual, np.load(DATA / filename) as expected:
                    for key in expected.files:
                        np.testing.assert_allclose(actual[key], expected[key], atol=1e-10, rtol=0)
            for source in DATA.glob("sensors_*.csv"):
                self.assertEqual((output / source.name).read_bytes(), source.read_bytes())
            self.assertFalse((Path(tmp) / "data").exists())

    def test_plot_refuses_an_existing_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "original.png"
            output.write_bytes(b"original")
            with self.assertRaises(FileExistsError):
                plot_overview.plot_overview(DATA, output)
            self.assertEqual(output.read_bytes(), b"original")

    def test_plot_and_audit_cli_from_another_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "overview.png"
            self.run_python(["-m", "classroom_heat_pinn_data.synthetic.plot_overview",
                             "--output", output], tmp)
            self.assertTrue(output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            result = self.run_python([ROOT / "classroom_heat_pinn_data/synthetic/audit.py",
                                     "--scan-points", "2"], tmp)
            self.assertIn("truth access:", result.stdout)
            self.assertIn("Jacobian rank 4/4", result.stdout)
            self.assertIn("bilinear sampling at original sensor coordinates", result.stdout)
            self.assertFalse((Path(tmp) / "data").exists())


if __name__ == "__main__":
    unittest.main()
