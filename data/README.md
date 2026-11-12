# Data

`dataset.csv` — 1,450 rows, 28 columns.

Source: NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set,
FD001 subset (Saxena & Goebel, 2008, NASA Ames Prognostics Data
Repository). Original file: `train_FD001.txt` (20,631 rows, 100
engines). This project uses only the first 1,450 rows, in file order
(engines 1–6 complete, first 146 cycles of engine 7).

How the two added columns were made (see `src/prepare_data.py`):
- `RUL` = last cycle of that engine in the FULL FD001 file minus the
  current cycle. Computed before taking the first 1,450 rows.
- `needs_maintenance` = 1 if `RUL <= 30`, else 0. 186 rows are 1,
  1,264 rows are 0.

Columns: `engine_id, cycle, os1, os2, os3, s1..s21, RUL, needs_maintenance`.
`os` = operational setting, `s` = sensor measurement. The dataset
readme does not name the individual sensors, so this project keeps
the original `s1..s21` names and does not invent sensor names.

Licence: NASA public data (NASA Open Government Licence, as stated
by the mirror). No personal data is involved; the data are simulated.
