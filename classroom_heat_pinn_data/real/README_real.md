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

Run: `python prepare_real.py` (expects `medellin/` with the .mat + Time*.xlsx files and `intel/` with data.txt.gz + mote_locs.txt).
