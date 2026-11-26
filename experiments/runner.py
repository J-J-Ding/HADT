from pathlib import Path
import copy
import csv
import json
import torch

from data_provider.data_factory import build_data_bundle
from utils.model_registry import build_model
from utils.channel_policy import resolve_policy
from utils.provenance import identity, atomic_json, sha256
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
    # Legacy HADT-only entry used by frozen experimental scripts.
    from utils.model_registry import build_hadt
    return build_hadt(config, pred_len)


def run_single_experiment(config, dataset_name, pred_len, seed):
    config = copy.deepcopy(config)
    config["model"].setdefault("quantiles", [0.05,0.1,0.25,0.5,0.75,0.9,0.95])
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
    named = "name" in config["model"]
    model = build_model(config, pred_len, bundle.n_vars).to(device)
    policy = resolve_policy(model, config)
    config["resolved_channel_policy"] = policy
    if named:
        run_dir = (output_root / config["experiment"]["run_name"] / model.model_name /
                   model.variant / dataset_name / f"H{pred_len}" / f"seed{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    signature, provenance = identity(config, dataset_name, pred_len, seed)
    marker = run_dir / "run_identity.json"
    if marker.exists():
        if json.loads(marker.read_text())["fingerprint"] != signature:
            raise ValueError(f"Run identity conflict; choose a new run_name: {run_dir}")
    elif named and any(run_dir.iterdir()):
        raise ValueError(f"Nonempty run without identity; refusing overwrite: {run_dir}")
    if named and (run_dir / "complete.json").exists():
        completed = json.loads((run_dir / "complete.json").read_text())
        for name, digest in completed["artifact_sha256"].items():
            if not (run_dir/name).is_file() or sha256(run_dir/name) != digest:
                raise ValueError(f"Completed artifact mismatch: {run_dir/name}")
        return json.loads((run_dir/"metrics.json").read_text())
    atomic_json(marker, dict(fingerprint=signature))
    atomic_json(run_dir / "source_manifest.json", provenance)


    logger = create_logger(run_dir / "train.log")
    logger.info(
        f"Starting dataset={dataset_name}, H={pred_len}, seed={seed}, "
        f"device={device}, C={bundle.n_vars}"
    )
    logger.info(f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES')} GPU={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")
    logger.info(f"Feature columns: {bundle.feature_columns}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    resolved = {
        "dataset": dataset_name,
        "pred_len": int(pred_len),
        "seed": int(seed),
        "n_vars": bundle.n_vars,
        "n_rows": bundle.n_rows,
        "feature_columns": bundle.feature_columns,
        "split_bounds": bundle.split_bounds,
        "model_name": model.model_name,
        "variant": model.variant,
        "output_mode": model.output_mode,
        "native_config": getattr(model,"native_config",None),
        "quantile_adapter": getattr(model,"quantile_adapter","native_features"),
        "head_input_dim": getattr(getattr(getattr(model,"quantile_head",None),"projection",None),"in_features",None),
        "n_patches": getattr(model,"n_patches",None),
        "n_points": getattr(model,"n_points",None),
        "head_dim": getattr(getattr(model,"decoder",None),"head_dim",None),
        "trainable_parameters": n_params,
        "config": config,
        "scaler": bundle.scaler.state_dict(),
        "runtime_versions": {"torch":torch.__version__,"cuda":torch.version.cuda},
        "device":str(device),
        "visible_devices":__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
        "scope":config.get('experiment',{}).get('scope','training_and_full_test'),
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
        channel_chunk_size=policy["evaluation_chunk_size"],
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
        "best_val_mse" if model.output_mode=="point" else "best_val_pinball": float(best_val),
        "model_name": model.model_name,
        "variant": model.variant,
        "output_mode": model.output_mode,
        "training_seconds": float(train_seconds),
        "trainable_parameters": int(n_params),
    })
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if config.get("visualization", {}).get("enabled", True):
        from engine.visualization import create_run_figures
        create_run_figures(model, bundle, device, run_dir, metrics,
                           max_channels=config.get("visualization", {}).get("max_channels"))
    artifacts = ["best_checkpoint.pt","metrics.json","resolved_config.json","training_history.csv","source_manifest.json","run_identity.json"]
    if config.get("visualization",{}).get("enabled",True):
        from scripts.plot_run_forecasts import render_run
        render_run(dict(source_dir=str(run_dir.resolve()),dataset=dataset_name,horizon=pred_len,
                        seed=seed,candidate_id=model.model_name+":"+model.variant))
        artifacts.extend(str(p.relative_to(run_dir)) for p in sorted((run_dir/"figures").rglob("*"))
                         if p.is_file() and p.suffix in (".png",".pdf",".html",".json",".npz"))
    if (run_dir/"horizon_metrics.csv").exists():
        artifacts.append("horizon_metrics.csv")
    artifact_hashes={name:sha256(run_dir/name) for name in artifacts}
    atomic_json(run_dir/"complete.json",dict(fingerprint=signature,scope=config.get("experiment",{}).get("scope","training_and_full_test"),model_name=model.model_name,
                variant=model.variant,output_mode=model.output_mode,dataset=dataset_name,
                horizon=pred_len,seed=seed,source_dir=str(run_dir.resolve()),
                candidate_id=model.model_name+":"+model.variant,artifact_sha256=artifact_hashes))
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
