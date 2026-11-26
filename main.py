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
    parser.add_argument("--model", choices=["hadt_nq","hadt_detail","patchtst","timemixer","dlinear","itransformer"])
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "smoke":
        if args.model is None and args.config == "configs/default.yaml":
            run_smoke_test()
        else:
            from utils.model_registry import build_model
            from utils.losses import forecasting_loss
            config = load_config(args.config)
            if args.model and args.model != config["model"].get("name","hadt_nq"):
                raise ValueError("--model must match the model in --config")
            model = build_model(config,args.pred_len,5)
            torch.set_num_threads(2)
            x = torch.randn(2,config["data"]["seq_len"],5)
            target = torch.randn(2,args.pred_len,5)
            prediction = model(x)
            loss = forecasting_loss(prediction,target)
            loss.backward()
            print(f"Smoke passed: {model.model_name}, output={tuple(prediction.shape)}, loss={loss.item():.6f}")
        return

    config = load_config(args.config)
    if args.model and args.model != config["model"].get("name","hadt_nq"):
        raise ValueError("--model must match --config")

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
