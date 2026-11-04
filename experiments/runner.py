from pathlib import Path
import copy
import csv
import json
import torch

from data_provider.data_factory import build_data_bundle
from models.hadt_nq import HADTNQ
from engine.trainer import Trainer
from engine.checkpoint import load_checkpoint
from engine.evaluator import evaluate_test
from utils.logger import create_logger
from utils.seed import set_seed


def _resolve_device(config):
    requested = config["runtime"].get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


def _build_model(config, pred_len):
    data_cfg = config["data"]
    model_cfg = config["model"]
    model = HADTNQ(
        seq_len=int(data_cfg["seq_len"]),
        pred_len=int(pred_len),
        patch_len=int(model_cfg["patch_len"]),
        stride=int(model_cfg["stride"]),
        d_model=int(model_cfg["d_model"]),
        n_heads=int(model_cfg["n_heads"]),
        e_layers=int(model_cfg["e_layers"]),
        d_ff=int(model_cfg["d_ff"]),
        dropout=float(model_cfg["dropout"]),
        n_points=model_cfg["n_points"],
        max_offset=float(model_cfg["max_offset"]),
        max_horizon=int(model_cfg["max_horizon"]),
        horizon_chunk_size=int(model_cfg["horizon_chunk_size"]),
        gradient_checkpointing=bool(model_cfg["gradient_checkpointing"]),
        min_quantile_gap=float(model_cfg["min_quantile_gap"]),
        quantiles=tuple(model_cfg["quantiles"]),
    )

    model.ablation = config.get("ablation")
    return model


def run_single_experiment(config, dataset_name, pred_len, seed):
    config = copy.deepcopy(config)
    if "torch_threads" in config["runtime"]:
        torch.set_num_threads(int(config["runtime"]["torch_threads"]))
    set_seed(seed)
    device = _resolve_device(config)

    bundle = build_data_bundle(config, dataset_name, pred_len)
    config["data_scaler"] = bundle.scaler.state_dict()
    config["feature_columns"] = bundle.feature_columns

    output_root = Path(config["experiment"]["output_dir"])
    run_dir = (
        output_root
        / config["experiment"]["run_name"]
        / dataset_name
        / f"H{pred_len}"
        / f"seed{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = create_logger(run_dir / "train.log")
    logger.info(
        f"Starting dataset={dataset_name}, H={pred_len}, seed={seed}, "
        f"device={device}, C={bundle.n_vars}"
    )
    logger.info(f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES')} GPU={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")
    logger.info(f"Feature columns: {bundle.feature_columns}")

    model = _build_model(config, pred_len).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    resolved = {
        "dataset": dataset_name,
        "pred_len": int(pred_len),
        "seed": int(seed),
        "n_vars": bundle.n_vars,
        "n_rows": bundle.n_rows,
        "feature_columns": bundle.feature_columns,
        "split_bounds": bundle.split_bounds,
        "n_patches": model.n_patches,
        "n_points": model.n_points,
        "head_dim": model.d_model // int(config["model"]["n_heads"]),
        "trainable_parameters": n_params,
        "config": config,
        "scaler": bundle.scaler.state_dict(),
    }
    with open(run_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(resolved, f, indent=2, ensure_ascii=False)

    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(train_cfg["scheduler_factor"]),
        patience=int(train_cfg["scheduler_patience"]),
        min_lr=float(train_cfg["min_learning_rate"]),
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        run_dir=run_dir,
        logger=logger,
    )

    checkpoint_path, best_val, train_seconds = trainer.fit(
        bundle.train_loader, bundle.val_loader
    )
    load_checkpoint(checkpoint_path, model, map_location=device)

    metrics = evaluate_test(
        model=model,
        loader=bundle.test_loader,
        device=device,
        scaler=bundle.scaler,
        channel_chunk_size=int(config["evaluation"]["channel_chunk_size"]),
        quantiles=tuple(config["model"]["quantiles"]),
        output_dir=run_dir,
        save_horizon_metrics=bool(
            config["evaluation"]["save_horizon_metrics"]
        ),
    )

    metrics.update({
        "dataset": dataset_name,
        "pred_len": int(pred_len),
        "seed": int(seed),
        "best_val_pinball": float(best_val),
        "training_seconds": float(train_seconds),
        "trainable_parameters": int(n_params),
    })
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if config.get("visualization", {}).get("enabled", True):
        from engine.visualization import create_run_figures
        create_run_figures(model, bundle, device, run_dir, metrics,
                           max_channels=config.get("visualization", {}).get("max_channels"))
    logger.info(f"Test metrics: {metrics}")
    return metrics


def run_experiment_matrix(config, dataset=None, pred_len=None, seed=None):
    datasets = [dataset] if dataset else list(config["data"]["datasets"])
    horizons = [int(pred_len)] if pred_len else list(config["experiment"]["horizons"])
    seeds = [int(seed)] if seed is not None else list(config["experiment"]["seeds"])

    all_results = []
    for dataset_name in datasets:
        for horizon in horizons:
            for run_seed in seeds:
                result = run_single_experiment(
                    config, dataset_name, horizon, run_seed
                )
                all_results.append(result)

    output_root = Path(config["experiment"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = (
        output_root / config["experiment"]["run_name"] / "summary.csv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if all_results:
        fieldnames = sorted({key for row in all_results for key in row.keys()})
        with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

    print(f"Experiment matrix complete. Summary: {summary_path}")
    return all_results
