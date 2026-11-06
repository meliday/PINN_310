"""Turn downloaded real datasets into CSVs in a separate output directory.

Importing this module does not read raw data or write files. Medellin requires
openpyxl; Intel accepts either intel/data.txt.gz or intel/data.txt.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR / "prepared"
MEDELLIN_OUTPUTS = ["medellin_sensor_coords.csv", "medellin_ref_offsets.csv",
                    "medellin_ext_outdoor.csv"] + [
    f"medellin_lab2_sample{k}_{kind}.csv"
    for k in ("02", "04") for kind in ("raw", "offsetcorrected")
]
INTEL_OUTPUTS = ["intel_temperature_30min.csv", "intel_mote_locs.csv"]


def read_times(k, source_dir=BASE_DIR / "medellin"):
    # Keep the optional Excel dependency out of Intel-only runs and imports.
    import openpyxl
    workbook = openpyxl.load_workbook(Path(source_dir) / f"Time{k}.xlsx", read_only=True)
    try:
        return pd.to_datetime([r[0] for r in workbook.active.iter_rows(min_row=2, values_only=True)])
    finally:
        workbook.close()


def load_sample(k, var="tem", source_dir=BASE_DIR / "medellin"):
    source_dir = Path(source_dir)
    T = sio.loadmat(source_dir / f"Sample{k}.mat", variable_names=[var])[var]
    if var == "tem":
        T = T / 100.0  # indoor values are centi-degrees; outdoor temE is already C
    t = read_times(k, source_dir)
    assert len(t) == T.shape[0], (k, len(t), T.shape)
    cols = [f"T{i:02d}" for i in range(1, T.shape[1] + 1)] if T.shape[1] > 1 else ["T_ext"]
    return pd.DataFrame(T, index=t, columns=cols)


def _prepare_medellin(source_dir, output_dir):
    source_dir = source_dir / "medellin"
    ref = load_sample("06", source_dir=source_dir)
    lab2 = {k: load_sample(k, source_dir=source_dir) for k in ("02", "04")}
    ext = load_sample("07", "temE", source_dir=source_dir)
    # Coordinates in metres, from Table 3 (LAB1 and LAB2 layouts).
    coords = pd.DataFrame(
        [(1, 7.21, 0.60, 5.70, 7.54), (2, 3.88, 5.88, 5.70, 10.17), (3, 5.85, 5.88, 3.30, 11.60),
         (4, 0.60, 0.60, 3.30, 9.12), (5, 1.92, 1.94, 3.30, 6.34), (6, 3.88, 3.91, 3.30, 3.86),
         (7, 1.92, 3.91, 3.30, 1.38), (8, 5.85, 3.91, 5.70, 4.91), (9, 7.21, 7.22, 5.70, 2.43),
         (10, 1.92, 5.88, 0.90, 2.43), (11, 0.60, 7.22, 0.90, 10.17), (12, 5.85, 1.94, 0.90, 4.91),
         (13, 3.88, 1.94, 0.90, 7.54)],
        columns=["sensor", "lab1_x_m", "lab1_y_m", "lab2_x_m", "lab2_y_m"])

    coords.to_csv(output_dir / "medellin_sensor_coords.csv", index=False)
    dev = ref.sub(ref.median(axis=1), axis=0)
    offsets = dev.median()
    offsets.rename("offset_C").to_csv(output_dir / "medellin_ref_offsets.csv")
    report = [f"REF offsets (C): min {offsets.min():+.2f}, max {offsets.max():+.2f}, "
              f"spread std {offsets.std():.2f}"]
    ext.to_csv(output_dir / "medellin_ext_outdoor.csv", index_label="time")
    for k, df in lab2.items():
        bad = (df < -5) | (df > 60) | df.isna()
        df = df.mask(bad)
        df.to_csv(output_dir / f"medellin_lab2_sample{k}_raw.csv", index_label="time")
        corr = df - offsets
        corr.to_csv(output_dir / f"medellin_lab2_sample{k}_offsetcorrected.csv", index_label="time")
        hourly = corr.resample("1h").mean()
        spread = hourly.max(axis=1) - hourly.min(axis=1)
        outdoor = ext["T_ext"].reindex(hourly.index, method="nearest", tolerance=pd.Timedelta("10min"))
        report.append(
            f"LAB2 Sample{k}: {df.index[0]} -> {df.index[-1]}, {len(df)} rows, "
            f"invalid {int(bad.values.sum())}; room mean {np.nanmean(corr.values):.2f} C; "
            f"hourly spatial spread (max-min) median {spread.median():.2f} C, max {spread.max():.2f} C; "
            f"outdoor during period {outdoor.min():.1f}-{outdoor.max():.1f} C "
            f"(coverage {outdoor.notna().mean()*100:.0f}%)"
        )
    return report


def intel_input_path(source_dir=BASE_DIR):
    directory = Path(source_dir) / "intel"
    for name in ("data.txt.gz", "data.txt"):
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Expected data.txt.gz or data.txt in {directory}")


def load_intel(source_dir=BASE_DIR):
    """Read, filter and aggregate Intel data without writing outputs."""
    source_dir = Path(source_dir)
    cols = ["date", "time", "epoch", "moteid", "temperature", "humidity", "light", "voltage"]
    d = pd.read_csv(intel_input_path(source_dir), sep=r"\s+", names=cols, header=None,
                    on_bad_lines="skip", engine="c", compression="infer")
    n0 = len(d)
    d = d.dropna(subset=["moteid", "temperature"])
    d = d[(d.moteid >= 1) & (d.moteid <= 54)].copy()
    # Both whole-second and fractional-second timestamps occur in the raw file.
    d["ts"] = pd.to_datetime(d["date"] + " " + d["time"], format="mixed", errors="coerce")
    invalid_times = int(d["ts"].isna().sum())
    d = d.dropna(subset=["ts"])
    ok = d.temperature.between(5, 45) & (d.voltage >= 2.3)
    report = [f"Intel: {n0} raw rows -> {len(d)} parsable; invalid timestamps {invalid_times}; "
              f"removed {int((~ok).sum())} outliers (T outside 5-45 C or voltage < 2.3 V)"]
    d = d[ok].copy()
    d["moteid"] = d["moteid"].astype(int)
    w = (d.set_index("ts").groupby("moteid")["temperature"].resample("30min").mean()
         .unstack(0).sort_index())
    w.columns = [f"m{c:02d}" for c in w.columns]
    locs = pd.read_csv(source_dir / "intel/mote_locs.txt", sep=r"\s+",
                       names=["mote", "x_m", "y_m"])
    night = w.between_time("02:00", "04:00")
    cov = night.notna().mean()
    report.append(
        f"Intel 30-min table: {w.shape[0]} times x {w.shape[1]} motes; motes with >=80% coverage at 02-04h: "
        f"{int((cov >= 0.8).sum())}; median night spatial spread "
        f"{(night.max(axis=1)-night.min(axis=1)).median():.2f} C"
    )
    return w, locs, report


def prepare(source_dir=BASE_DIR, output_dir=DEFAULT_OUTPUT, dataset="all"):
    """Prepare selected datasets, refusing to replace any existing output file."""
    if dataset not in ("all", "medellin", "intel"):
        raise ValueError("dataset must be all, medellin or intel")
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    expected = ["REPORT.txt"]
    if dataset in ("all", "medellin"):
        expected += MEDELLIN_OUTPUTS
    if dataset in ("all", "intel"):
        expected += INTEL_OUTPUTS
    if any((output_dir / name).exists() for name in expected):
        raise FileExistsError(f"Existing outputs in {output_dir}; choose a new --output-dir")
    # Check the full input set before writing even the first output.
    inputs = []
    if dataset in ("all", "medellin"):
        inputs += [source_dir / "medellin" / f"{prefix}{k}.{suffix}"
                   for k in ("02", "04", "06", "07")
                   for prefix, suffix in (("Sample", "mat"), ("Time", "xlsx"))]
    if dataset in ("all", "intel"):
        inputs += [intel_input_path(source_dir), source_dir / "intel/mote_locs.txt"]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = []
    if dataset in ("all", "medellin"):
        report.extend(_prepare_medellin(source_dir, output_dir))
    if dataset in ("all", "intel"):
        w, locs, intel_report = load_intel(source_dir)
        w.to_csv(output_dir / "intel_temperature_30min.csv", index_label="time")
        locs.to_csv(output_dir / "intel_mote_locs.csv", index=False)
        report.extend(intel_report)
    (output_dir / "REPORT.txt").write_text("\n".join(report) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="New output directory (default: real/prepared)")
    parser.add_argument("--dataset", choices=("all", "medellin", "intel"), default="all")
    args = parser.parse_args(argv)
    print("\n".join(prepare(args.source_dir, args.output_dir, args.dataset)))


if __name__ == "__main__":
    main()
