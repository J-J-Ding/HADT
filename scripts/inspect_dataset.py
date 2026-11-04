import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
from data_provider.csv_dataset import read_multivariate_csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--time_column", default="date")
    args = parser.parse_args()

    values, columns = read_multivariate_csv(args.csv, args.time_column)
    print(f"File: {Path(args.csv).resolve()}")
    print(f"Rows T: {values.shape[0]}")
    print(f"Variables C: {values.shape[1]}")
    print(f"Features: {columns}")
    print(f"Finite: {bool(__import__('numpy').isfinite(values).all())}")

if __name__ == "__main__":
    main()
