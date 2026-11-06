"""Turn the downloaded real datasets into clean CSVs.

Medellín ITM (Botero-Valencia et al., Data 2023, OSF zbn8w, CC BY 4.0)
  LAB2: Sample02, Sample04  (13 sensors, 10 s)
  REF : Sample06            (all 13 sensors side by side -> calibration offsets)
  EXT : Sample07            (outdoor temperature, 4 min)
Intel Berkeley Lab (2004): data.txt.gz + mote_locs.txt
"""
import gzip
import os

import numpy as np
import openpyxl
import pandas as pd
import scipy.io as sio

M = "medellin"
OUT = "clean"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- Medellín
# Sensor coordinates in metres, from Table 3 of the paper (LAB1, LAB2)
coords = pd.DataFrame(
    [(1, 7.21, 0.60, 5.70, 7.54), (2, 3.88, 5.88, 5.70, 10.17), (3, 5.85, 5.88, 3.30, 11.60),
     (4, 0.60, 0.60, 3.30, 9.12), (5, 1.92, 1.94, 3.30, 6.34), (6, 3.88, 3.91, 3.30, 3.86),
     (7, 1.92, 3.91, 3.30, 1.38), (8, 5.85, 3.91, 5.70, 4.91), (9, 7.21, 7.22, 5.70, 2.43),
     (10, 1.92, 5.88, 0.90, 2.43), (11, 0.60, 7.22, 0.90, 10.17), (12, 5.85, 1.94, 0.90, 4.91),
     (13, 3.88, 1.94, 0.90, 7.54)],
    columns=["sensor", "lab1_x_m", "lab1_y_m", "lab2_x_m", "lab2_y_m"])
coords.to_csv(f"{OUT}/medellin_sensor_coords.csv", index=False)


def read_times(k):
    ws = openpyxl.load_workbook(f"{M}/Time{k}.xlsx", read_only=True).active
    return pd.to_datetime([r[0] for r in ws.iter_rows(min_row=2, values_only=True)])


def load_sample(k, var="tem"):
    m = sio.loadmat(f"{M}/Sample{k}.mat")
    T = m[var]
    if var == "tem":
        T = T / 100.0          # indoor sensors are stored in centi-degrees C (e.g. 2450 = 24.50 C)
    t = read_times(k)
    assert len(t) == T.shape[0], (k, len(t), T.shape)
    cols = [f"T{i:02d}" for i in range(1, T.shape[1] + 1)] if T.shape[1] > 1 else ["T_ext"]
    return pd.DataFrame(T, index=t, columns=cols)


report = []
ref = load_sample("06")
lab2 = {k: load_sample(k) for k in ("02", "04")}
ext = load_sample("07", "temE")

# calibration offsets from REF (sensors co-located): offset_i = median(T_i - median over sensors)
dev = ref.sub(ref.median(axis=1), axis=0)
offsets = dev.median()
offsets.rename("offset_C").to_csv(f"{OUT}/medellin_ref_offsets.csv")
report.append(f"REF offsets (C): min {offsets.min():+.2f}, max {offsets.max():+.2f}, "
              f"spread std {offsets.std():.2f}")

ext.to_csv(f"{OUT}/medellin_ext_outdoor.csv", index_label="time")
for k, df in lab2.items():
    bad = (df < -5) | (df > 60) | df.isna()
    df = df.mask(bad)
    df.to_csv(f"{OUT}/medellin_lab2_sample{k}_raw.csv", index_label="time")
    corr = df - offsets
    corr.to_csv(f"{OUT}/medellin_lab2_sample{k}_offsetcorrected.csv", index_label="time")
    h = corr.resample("1h").mean()
    spread = (h.max(axis=1) - h.min(axis=1))
    e = ext["T_ext"].reindex(h.index, method="nearest", tolerance=pd.Timedelta("10min"))
    report.append(
        f"LAB2 Sample{k}: {df.index[0]} -> {df.index[-1]}, {len(df)} rows, invalid {int(bad.values.sum())}; "
        f"room mean {np.nanmean(corr.values):.2f} C; hourly spatial spread (max-min) median {spread.median():.2f} C, "
        f"max {spread.max():.2f} C; outdoor during period {e.min():.1f}-{e.max():.1f} C (coverage {e.notna().mean()*100:.0f}%)")

# ---------------------------------------------------------------- Intel
cols = ["date", "time", "epoch", "moteid", "temperature", "humidity", "light", "voltage"]
with gzip.open("intel/data.txt.gz", "rt") as f:
    d = pd.read_csv(f, sep=r"\s+", names=cols, header=None, on_bad_lines="skip", engine="c")
n0 = len(d)
d = d.dropna(subset=["moteid", "temperature"])
d = d[(d.moteid >= 1) & (d.moteid <= 54)]
d["ts"] = pd.to_datetime(d["date"] + " " + d["time"], errors="coerce")
d = d.dropna(subset=["ts"])
ok = d.temperature.between(5, 45) & (d.voltage >= 2.3)
report.append(f"Intel: {n0} raw rows -> {len(d)} parsable; removed {int((~ok).sum())} outliers "
              f"(T outside 5-45 C or voltage < 2.3 V)")
d = d[ok]
d["moteid"] = d["moteid"].astype(int)
w = (d.set_index("ts").groupby("moteid")["temperature"].resample("30min").mean()
     .unstack(0).sort_index())
w.columns = [f"m{c:02d}" for c in w.columns]
w.to_csv(f"{OUT}/intel_temperature_30min.csv", index_label="time")
locs = pd.read_csv("intel/mote_locs.txt", sep=r"\s+", names=["mote", "x_m", "y_m"])
locs.to_csv(f"{OUT}/intel_mote_locs.csv", index=False)
night = w.between_time("02:00", "04:00")
cov = night.notna().mean()
report.append(f"Intel 30-min table: {w.shape[0]} times x {w.shape[1]} motes; motes with >=80% coverage at 02-04h: "
              f"{int((cov >= 0.8).sum())}; median night spatial spread {(night.max(axis=1)-night.min(axis=1)).median():.2f} C")

with open(f"{OUT}/REPORT.txt", "w") as f:
    f.write("\n".join(report) + "\n")
print("\n".join(report))
