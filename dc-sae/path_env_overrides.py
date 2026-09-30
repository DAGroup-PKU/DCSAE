from __future__ import annotations

import os
from typing import Any


SAE_CONFIG_PATH_ENV_VARS = {
    "data.train_path": "DECO_DATA_TRAIN_PATH",
    "data.val_path": "DECO_DATA_VAL_PATH",
    "encoder.dinov3_model_dir": "DECO_DINOV3_MODEL_DIR",
    "encoder.siglip2_model_name": "DECO_SIGLIP2_MODEL_NAME",
    "encoder.dinov2_model_name": "DECO_DINOV2_MODEL_NAME",
    "encoder.qwen3_vit_model_name": "DECO_QWEN3_VIT_MODEL_NAME",
    "encoder.internvl3_model_name": "DECO_INTERNVL3_MODEL_NAME",
    "discriminator.dino_ckpt_path": "DECO_DINO_CKPT_PATH",
    "logging.output_dir": "DC_SAE_OUTPUT_DIR",
    "checkpoint.sae_ckpt": "DC_SAE_CKPT_PATH",
}

DIT_RUNTIME_PATH_ENV_VARS = {
    "data_path": "DECO_DATA_TRAIN_PATH",
    "results_dir": "DECO_DIT_RESULTS_DIR",
    "sae_ckpt": "DECO_DIT_SAE_CKPT_PATH",
    "resume_ckpt": "DECO_DIT_RESUME_CKPT_PATH",
    "fid_ref_path": "DECO_FID_REF_PATH",
    "ref_images_path": "DECO_REF_IMAGES_PATH",
    "latent_stats_path": "DECO_LATENT_STATS_PATH",
}

DIT_CONFIG_PATH_ENV_VARS = {
    "misc.latent_stats_path": "DECO_LATENT_STATS_PATH",
}

CLUSTER_PATH_ENV_VARS = {
    "conda_path": "DECO_CONDA_PATH",
    "workspace": "DECO_WORKSPACE",
}

SAE_LAUNCH_ARG_PATH_ENV_VARS = {
    "output_dir": "DC_SAE_OUTPUT_DIR",
    "sae_ckpt": "DC_SAE_CKPT_PATH",
    "conda_path": "DECO_CONDA_PATH",
}

DIT_TRAIN_ARG_PATH_ENV_VARS = {
    "data_path": "DECO_DATA_TRAIN_PATH",
    "results_dir": "DECO_DIT_RESULTS_DIR",
    "sae_ckpt": "DECO_DIT_SAE_CKPT_PATH",
    "ckpt": "DECO_DIT_RESUME_CKPT_PATH",
    "fid_ref_path": "DECO_FID_REF_PATH",
    "ref_images_path": "DECO_REF_IMAGES_PATH",
    "latent_stats_path": "DECO_LATENT_STATS_PATH",
}

TRAIN_VAE_ARG_PATH_ENV_VARS = {
    "train_path": "DECO_DATA_TRAIN_PATH",
    "val_path": "DECO_DATA_VAL_PATH",
    "output_dir": "DECO_VAE_OUTPUT_DIR",
    "vae_ckpt": "DECO_VAE_CKPT_PATH",
    "dinov3_dir": "DECO_DINOV3_MODEL_DIR",
    "siglip2_model_name": "DECO_SIGLIP2_MODEL_NAME",
    "dinov2_model_name": "DECO_DINOV2_MODEL_NAME",
    "dino_ckpt_path": "DECO_DINO_CKPT_PATH",
}


def apply_path_env_overrides(
    target: Any,
    field_to_env: dict[str, str],
    *,
    enabled: bool,
) -> list[dict[str, Any]]:
    overrides: list[dict[str, Any]] = []
    if not enabled:
        return overrides

    for field_name, env_name in field_to_env.items():
        env_value = os.environ.get(env_name)
        if env_value is None or env_value == "":
            continue

        previous = get_nested_value(target, field_name)
        set_nested_value(target, field_name, env_value)
        overrides.append(
            {
                "field": field_name,
                "env": env_name,
                "value": env_value,
                "previous": previous,
            }
        )

    return overrides


def get_nested_value(target: Any, field_name: str) -> Any:
    current = target
    for part in field_name.split("."):
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
            continue

        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def set_nested_value(target: Any, field_name: str, value: Any) -> None:
    parts = field_name.split(".")
    current = target

    for part in parts[:-1]:
        if isinstance(current, dict):
            next_value = current.get(part)
            if next_value is None:
                next_value = {}
                current[part] = next_value
            elif not isinstance(next_value, dict):
                raise TypeError(f"Cannot descend into non-mapping field `{part}` for `{field_name}`.")
            current = next_value
            continue

        if not hasattr(current, part):
            raise AttributeError(f"Object of type {type(current)!r} has no attribute `{part}`.")
        current = getattr(current, part)

    last = parts[-1]
    if isinstance(current, dict):
        current[last] = value
        return

    if not hasattr(current, last):
        raise AttributeError(f"Object of type {type(current)!r} has no attribute `{last}`.")
    setattr(current, last, value)


def format_path_env_overrides(
    scope: str,
    overrides: list[dict[str, Any]],
) -> list[str]:
    lines: list[str] = []
    for item in overrides:
        lines.append(f"  [{scope}] {item['field']} <= ${item['env']} -> {item['value']}")
    return lines
