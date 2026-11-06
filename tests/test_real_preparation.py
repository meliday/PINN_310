"""Regression checks for raw-file selection, timestamps and unit conversion."""

import gzip
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import scipy.io as sio

from classroom_heat_pinn_data.real import prepare_real


INTEL_SAMPLE = """2004-02-28 02:00:00 1 1 20 40 10 2.8
2004-02-28 02:10:00.500000 2 1 22 40 10 2.8
2004-02-28 02:15:00 3 1 122 40 10 2.0
invalid-date 02:20:00 4 1 21 40 10 2.8
"""


class RealPreparationTests(unittest.TestCase):
    def make_intel_input(self, directory, compressed=False):
        source = Path(directory) / "source"
        intel = source / "intel"
        intel.mkdir(parents=True)
        (intel / "mote_locs.txt").write_text("1 2.0 3.0\n")
        if compressed:
            with gzip.open(intel / "data.txt.gz", "wt") as f:
                f.write(INTEL_SAMPLE)
        else:
            (intel / "data.txt").write_text(INTEL_SAMPLE)
        return source

    def test_plain_text_and_gzip_keep_both_timestamp_formats(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed), tempfile.TemporaryDirectory() as tmp:
                source = self.make_intel_input(tmp, compressed)
                w, locs, report = prepare_real.load_intel(source)
                self.assertEqual(w.shape, (1, 1))
                self.assertEqual(w.columns.tolist(), ["m01"])
                self.assertEqual(w.iloc[0, 0], 21.0)
                self.assertEqual(locs["mote"].tolist(), [1])
                self.assertIn("invalid timestamps 1", report[0])
                self.assertIn("removed 1 outliers", report[0])

    def test_intel_only_does_not_require_openpyxl(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_intel_input(tmp)
            output = Path(tmp) / "prepared"
            with mock.patch.dict("sys.modules", {"openpyxl": None}):
                prepare_real.prepare(source, output, dataset="intel")
            self.assertEqual(pd.read_csv(output / "intel_temperature_30min.csv")["m01"].iloc[0], 21)
            self.assertEqual(sorted(p.name for p in output.iterdir()),
                             ["REPORT.txt", "intel_mote_locs.csv", "intel_temperature_30min.csv"])

    def test_existing_clean_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = Path(tmp) / "intel_temperature_30min.csv"
            sentinel.write_text("original")
            with self.assertRaises(FileExistsError):
                prepare_real.prepare(Path(tmp) / "missing-source", tmp, dataset="intel")
            self.assertEqual(sentinel.read_text(), "original")
            self.assertEqual(list(Path(tmp).iterdir()), [sentinel])

    def test_missing_input_is_detected_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "prepared"
            with self.assertRaises(FileNotFoundError):
                prepare_real.prepare(Path(tmp) / "missing-source", output, dataset="all")
            self.assertFalse(output.exists())

    def test_medellin_units_offsets_and_outdoor_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            medellin = source / "medellin"
            medellin.mkdir(parents=True)
            offsets = np.linspace(-0.06, 0.06, 13)
            raw_C = np.tile(24 + offsets, (2, 1))
            for k in ("02", "04", "06", "07"):
                if k == "07":
                    values = {"temE": np.array([[17.5], [18.0]])}
                else:
                    values = {"tem": (np.tile(20 + offsets, (2, 1)) if k == "06" else raw_C) * 100}
                sio.savemat(medellin / f"Sample{k}.mat", values)
                (medellin / f"Time{k}.xlsx").touch()
            times = pd.date_range("2022-10-24 11:00", periods=2, freq="10s")
            output = Path(tmp) / "prepared"
            # The Excel reader is isolated here; all MAT, scaling and CSV code runs.
            with mock.patch.object(prepare_real, "read_times", return_value=times):
                prepare_real.prepare(source, output, dataset="medellin")
            for k in ("02", "04"):
                raw = pd.read_csv(output / f"medellin_lab2_sample{k}_raw.csv", index_col="time")
                corrected = pd.read_csv(output / f"medellin_lab2_sample{k}_offsetcorrected.csv", index_col="time")
                np.testing.assert_allclose(raw.to_numpy(), raw_C, atol=1e-12, rtol=0)
                np.testing.assert_allclose(corrected.to_numpy(), 24, atol=1e-12, rtol=0)
            outdoor = pd.read_csv(output / "medellin_ext_outdoor.csv")["T_ext"].to_numpy()
            np.testing.assert_array_equal(outdoor, [17.5, 18.0])


if __name__ == "__main__":
    unittest.main()
