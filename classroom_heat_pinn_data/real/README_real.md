# Real indoor-temperature data (cleaned)

## Medellín ITM (Botero-Valencia et al., Data 8(5):82, 2023; OSF 10.17605/OSF.IO/ZBN8W; CC BY 4.0)
- 13 calibrated Xiaomi sensors hung from the ceiling at one height; coordinates from the paper's Table 3 -> `clean/medellin_sensor_coords.csv` (metres; LAB1 and LAB2 layouts).
- Downloaded: Sample02 and Sample04 (LAB2, 56 h and 65 h, 10 s), Sample06 (REF: all sensors side by side), Sample07 (EXT: outdoor temperature, 4 min, Jun 2022 to Feb 2023).
- Not downloaded yet: Sample00/01 (LAB1, 33 and 45 MB) and Sample03/05 (LAB2, 19 and 30 MB).
- Units: the indoor .mat values are centi-degrees (2450 = 24.50 °C); the scripts divide by 100. Outdoor values are already °C.
- REF offsets are ±0.1 °C, so the sensors agree well. `*_offsetcorrected.csv` subtracts them.
- LAB2 is nearly uniform: hourly spatial spread (max − min) median 0.79 °C (Sample02) and 0.51 °C (Sample04). Room mean 23.7 °C while outdoors ranged 17–29 °C. The room looks actively conditioned (not confirmed in the paper excerpt we read).

## Intel Berkeley Lab (2004, MIT CSAIL)
- `clean/intel_temperature_30min.csv`: 30-min means per mote, after removing T outside 5–45 °C or voltage < 2.3 V (519,051 of 2.3 M rows).
- `clean/intel_mote_locs.csv`: mote x, y in metres.
- At 02–04 h, 24 motes have ≥ 80 % coverage; the median spatial spread is 3.1 °C.

The checked-in `clean/` files are preserved snapshots. From the repository root:

```bash
# Intel only: NumPy, SciPy, pandas >= 2; no openpyxl required.
python -B -m classroom_heat_pinn_data.real.prepare_real --dataset intel --output-dir /tmp/heat-intel-new
# All data: additionally requires openpyxl for Time*.xlsx.
python -B -m classroom_heat_pinn_data.real.prepare_real --dataset all --output-dir /tmp/heat-real-new
```

Use unused output paths. The default is `real/prepared/`, separate from `clean/`.
Existing output files cause an error before processing; they are not overwritten.
`--source-dir` can select a different root containing `medellin/` and `intel/`.
Default paths are based on the script location, so direct script execution also
works from other working directories. Importing the module performs no data I/O.

Intel accepts `data.txt.gz` or uncompressed `data.txt` (preferring `.gz` if both
are present), alongside `mote_locs.txt`. Mixed whole-second/fractional-second
timestamps are parsed explicitly. The previous parser dropped four valid
timestamp rows in the supplied raw file; new outputs can therefore differ from
the preserved `clean/` snapshot. This is a preprocessing correction, not a change
to the raw observations. See `../CODE_FIXES.md` for validation and limitations.
