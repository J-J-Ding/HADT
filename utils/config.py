from copy import deepcopy
from pathlib import Path
import yaml


EXPECTED_QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config


def clone_config(config):
    return deepcopy(config)


def validate_config(config):
    data = config["data"]
    model = config["model"]
    training = config["training"]
    experiment = config["experiment"]

    l = int(data["seq_len"])
    if l <= 0:
        raise ValueError("seq_len must be positive")
    from utils.model_registry import MODEL_NAMES, EXTERNAL_MODELS
    from utils.baseline_config import native_config
    name = model.get("name", "hadt_nq")
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}")
    if model.get("output_mode","quantile") not in ("point","quantile"):
        raise ValueError("output_mode must be point or quantile")
    if model.get("quantile_head","median_positive_gap") != "median_positive_gap":
        raise ValueError("Only the existing median-positive-gap head is supported")
    if model.get("quantile_adapter","post_point_shared_head") != "post_point_shared_head":
        raise ValueError("External quantile models require the post-point shared head")
    if model.get("quantiles", EXPECTED_QUANTILES) != EXPECTED_QUANTILES:
        raise ValueError(f"Quantiles must equal {EXPECTED_QUANTILES}")
    if name in EXTERNAL_MODELS:
        allowed={"name","output_mode","quantile_head","quantile_adapter","quantiles","variant","params"}
        if set(model)-allowed:
            raise ValueError(f"Unrecognized external model keys: {sorted(set(model)-allowed)}")
        for horizon in experiment["horizons"]:
            native_config(config, horizon, 1)
    else:
        from models.hadt_nq import HADTNQ
        import inspect
        params=model.get("params",model)
        if "params" in model:
            allowed=set(inspect.signature(HADTNQ).parameters)-{"seq_len","pred_len"}
            if name == "hadt_detail":
                from models.hadt_detail import DetailHADTNQ
                allowed |= set(inspect.signature(DetailHADTNQ).parameters)-{"kwargs"}
            if set(params)-allowed-{"encoder_kind"}:
                raise ValueError("Unknown HADT params")
        p=int(params.get("patch_len",16)); stride=int(params.get("stride",8))
        d=int(params.get("d_model",128)); heads=int(params.get("n_heads",8))
        if p<=0 or stride<=0 or p>l:
            raise ValueError("Require 0 < patch_len <= seq_len and stride > 0")
        if heads<=0 or d<=0 or d%heads:
            raise ValueError("d_model must be divisible by positive n_heads")
        if int(params.get("max_horizon",720)) < max(experiment["horizons"]):
            raise ValueError("max_horizon must cover all configured horizons")
    if training["batch_size"] <= 0:
        raise ValueError("batch_size must be positive.")

    ratios_cfg = data["split_ratios"]
    for group in ("ETT", "other"):
        ratios = ratios_cfg[group]
        if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-8:
            raise ValueError(f"split_ratios[{group}] must contain three values summing to 1.")
