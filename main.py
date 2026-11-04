import argparse
import torch

from utils.config import load_config
from utils.smoke_test import run_smoke_test
from experiments.runner import run_experiment_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="HADT-NQ independent time-series interval forecasting project"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--mode",
        choices=["smoke", "single", "experiment"],
        default="smoke",
    )
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "smoke":
        run_smoke_test()
        return

    config = load_config(args.config)

    if args.mode == "single":
        run_experiment_matrix(
            config,
            dataset=args.dataset,
            pred_len=args.pred_len,
            seed=args.seed,
        )
    elif args.mode == "experiment":
        run_experiment_matrix(config)


if __name__ == "__main__":
    main()
