"""Rebuild data/dataset.csv from the original NASA C-MAPSS FD001 file.

This script is documentation of exactly how dataset.csv was made.
You do NOT need to run it to run the project: dataset.csv is already
included, and train.py reads dataset.csv directly.

Steps (in order):
  1. Read train_FD001.txt (20,631 rows, 100 engines, 26 columns, no header).
  2. For every engine, find its last cycle (the cycle at which it failed).
     RUL = last_cycle - current_cycle.  This MUST be done on the full
     file, BEFORE taking the first 1,450 rows, because engine 7 is cut
     off at row 1,450 and its last cycle (259) is not in the sample.
  3. Keep only the first 1,450 rows, in file order.
     That gives engines 1-6 complete, plus the first 146 cycles of
     engine 7. It is a plain, reproducible rule: "first 1,450 rows".
  4. Label: needs_maintenance = 1 if RUL <= 30 cycles, else 0.
     30 cycles is the maintenance window chosen for this project.

Usage:
    python src/prepare_data.py --raw path/to/train_FD001.txt --out data/dataset.csv

Original source (free, public):
    Saxena, A. and Goebel, K. (2008). Turbofan Engine Degradation
    Simulation Data Set. NASA Ames Prognostics Data Repository.
    Mirror used here: https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
    (file inside the zip: CMAPSSData/train_FD001.txt)
"""

import argparse

import pandas as pd

COLUMNS = (["engine_id", "cycle", "os1", "os2", "os3"]
           + [f"s{i}" for i in range(1, 22)])
N_ROWS = 1450
MAINTENANCE_WINDOW = 30


def build_dataset(raw_path: str) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, sep=r"\s+", header=None, names=COLUMNS)
    last_cycle = raw.groupby("engine_id")["cycle"].max()
    raw["RUL"] = raw["engine_id"].map(last_cycle) - raw["cycle"]
    sample = raw.head(N_ROWS).copy()
    sample["needs_maintenance"] = (sample["RUL"] <= MAINTENANCE_WINDOW).astype(int)
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild dataset.csv from raw C-MAPSS FD001.")
    parser.add_argument("--raw", default="work/cmapss/train_FD001.txt")
    parser.add_argument("--out", default="data/dataset.csv")
    args = parser.parse_args()
    df = build_dataset(args.raw)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {len(df)} rows, "
          f"{int(df['needs_maintenance'].sum())} need maintenance, "
          f"{int((df['needs_maintenance'] == 0).sum())} do not.")


if __name__ == "__main__":
    main()
