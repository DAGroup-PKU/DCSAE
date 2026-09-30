# https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Short-Caption
"""
Train DiT (Diffusion Transformer) on dc-sae latent space.
Uses config-driven approach with minimal argparse.
Supports multi-node multi-GPU distributed training.
"""

import argparse
import gc
import math
import os
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from glob import glob
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
from torchvision import transforms

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

# FID imports
try:
    from pytorch_fid import fid_score
    HAS_FID = True
except ImportError:
    HAS_FID = False
    print("Warning: 'pytorch-fid' not found. FID calculation will be skipped.")

# Inception Score and Precision/Recall imports
try:
    from torch_fidelity import calculate_metrics
    HAS_TORCH_FIDELITY = True
except ImportError:
    HAS_TORCH_FIDELITY = False
    print("Warning: 'torch-fidelity' not found. IS and Recall calculations will be skipped.")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config_utils import load_config
from model import DCSAE
from models.rae.utils.ddp_utils import (
    cleanup_ddp,
    clip_grad_norm,
    create_logger,
    get_model_state_dict,
    get_optimizer_state_dict,
    requires_grad,
    setup_ddp,
    unwrap_model,
    update_ema,
    wrap_model_ddp,
    wrap_model_fsdp,
)
from models.rae.utils.image_utils import center_crop_arr
from models.rae.utils.model_utils import instantiate_from_config
from models.rae.utils.optim_utils import build_optimizer, build_scheduler
from models.rae.utils.train_utils import parse_configs
from models.rae.utils.vae_utils import LatentNormalizer, load_latent_stats
from path_env_overrides import (
    DIT_TRAIN_ARG_PATH_ENV_VARS,
    SAE_CONFIG_PATH_ENV_VARS,
    apply_path_env_overrides,
)


#################################################################################
#                              Training Utilities                               #
#################################################################################

def build_wrapped_dit_model(model: torch.nn.Module, training_cfg: dict, local_rank: int, precision: str):
    distributed_mode = str(training_cfg.get("distributed", "ddp")).lower()
    if distributed_mode == "fsdp":
        return wrap_model_fsdp(
            model,
            device_id=local_rank,
            use_bf16=precision == "bf16",
            sharding_strategy=str(training_cfg.get("fsdp_sharding_strategy", "FULL_SHARD")),
            use_orig_params=bool(training_cfg.get("fsdp_use_orig_params", True)),
        )
    return wrap_model_ddp(model, device_id=local_rank, gradient_as_bucket_view=False)


def sample_timesteps(
    batch_size: int,
    device: torch.device,
    time_shift: float = 1.0,
):
    """Sample timesteps for flow matching training with SD3-style time shift."""
    t = torch.rand(batch_size, device=device)
    t = (time_shift * t) / (1.0 + (time_shift - 1.0) * t)
    t = t.clamp(1e-5, 1.0 - 1e-5)
    return t


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _tensor_stat_triplet(tensor: torch.Tensor) -> tuple[float, float, float]:
    tensor = tensor.detach().float()
    mean = tensor.mean().item()
    std = tensor.std(unbiased=False).item()
    abs_mean = tensor.abs().mean().item()
    return mean, std, abs_mean


def _parse_step_checkpoint_name(path: str) -> int | None:
    stem = os.path.splitext(os.path.basename(path))[0]
    if not stem.isdigit():
        return None
    return int(stem)


def find_auto_resume_checkpoint(results_dir: str) -> tuple[str | None, str | None]:
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    numbered_checkpoints = []

    for path in glob(os.path.join(checkpoint_dir, "*.pt")):
        if os.path.abspath(path) == os.path.abspath(latest_path):
            continue
        step = _parse_step_checkpoint_name(path)
        if step is not None:
            numbered_checkpoints.append((step, path))

    numbered_checkpoints.sort(key=lambda item: item[0], reverse=True)
    newest_numbered = numbered_checkpoints[0][1] if numbered_checkpoints else None

    if os.path.isfile(latest_path):
        if newest_numbered and os.path.getmtime(newest_numbered) > os.path.getmtime(latest_path):
            reason = (
                f"`latest.pt` is older than numbered checkpoint "
                f"`{os.path.basename(newest_numbered)}`; using the newer numbered checkpoint."
            )
            return newest_numbered, reason
        return latest_path, "Using `checkpoints/latest.pt`."

    if newest_numbered:
        return newest_numbered, f"`latest.pt` not found; using `{os.path.basename(newest_numbered)}`."

    return None, "No checkpoint found under results_dir/checkpoints."


def compute_train_loss(
    model,
    x_latent: torch.Tensor,
    model_kwargs: dict,
    time_shift: float = 1.0,
    semantic_channels: int = None,
    component_split_ch: int = None,
    debug_context: dict | None = None,
):
    """
    Compute flow matching training loss.
    
    x_t = t * noise + (1 - t) * x_0
    target = x_0 (x-prediction)
    
    Args:
        model: DiT model
        x_latent: Latent tensor [B, C, H, W]
        model_kwargs: Additional model arguments
        time_shift: Time shift for flow matching
        semantic_channels: If not None, only compute loss on first semantic_channels channels
                          (used for zero_hf_mode to exclude HF channels from loss)
        component_split_ch: If not None, also return separate semantic/hf losses
                           split at this channel index (for joint mode monitoring)
    """
    B = x_latent.size(0)
    device = x_latent.device

    noise = torch.randn_like(x_latent)
    t = sample_timesteps(B, device, time_shift)

    t_broadcast = t.view(B, 1, 1, 1)
    x_t = t_broadcast * noise + (1.0 - t_broadcast) * x_latent

    model_output = model(x_t, t, **model_kwargs)

    reweight_scale = torch.clamp_max(1.0 / (t**2 + 1e-8), 10)
    rw = reweight_scale.view(B, 1, 1, 1)

    components = {}
    if component_split_ch is not None and component_split_ch > 0:
        sem_loss = F.mse_loss(
            model_output[:, :component_split_ch],
            x_latent[:, :component_split_ch],
            reduction="none",
        )
        sem_loss = sem_loss * rw
        components["semantic_loss"] = sem_loss.mean()
        hf_channels = x_latent.shape[1] - component_split_ch
        if hf_channels > 0:
            hf_loss = F.mse_loss(
                model_output[:, component_split_ch:],
                x_latent[:, component_split_ch:],
                reduction="none",
            )
            hf_loss = hf_loss * rw
            components["hf_loss"] = hf_loss.mean()
        else:
            hf_loss = None
            components["hf_loss"] = torch.zeros((), device=device, dtype=model_output.dtype)

        if semantic_channels is not None and semantic_channels > 0:
            # Keep zero_hf semantics: optimize only semantic channels.
            loss = components["semantic_loss"]
        else:
            # Use per-element merged mean so scale matches full-channel MSE.
            if hf_loss is not None:
                loss = torch.cat([sem_loss, hf_loss], dim=1).mean()
            else:
                loss = sem_loss.mean()
    else:
        if semantic_channels is not None and semantic_channels > 0:
            loss = F.mse_loss(model_output[:, :semantic_channels], x_latent[:, :semantic_channels], reduction="none")
        else:
            loss = F.mse_loss(model_output, x_latent, reduction="none")
        loss = (loss * rw).mean()

    return loss, model_output, noise, t, components


@torch.no_grad()
def sample_latent(
    model,
    batch_size: int,
    latent_shape: tuple,
    device: torch.device,
    y: torch.Tensor,
    time_shift: float,
    steps: int = 50,
    use_cfg: bool = False,
    cfg_scale: float = 3.0,
    null_class: int = 1000,
):
    """Sample latent z0 using Euler method."""
    model.eval()

    C, H, W = latent_shape
    x = torch.randn((batch_size, C, H, W), device=device, dtype=torch.float32)

    shift = float(time_shift)

    def flow_shift(t_lin: torch.Tensor) -> torch.Tensor:
        t = (shift * t_lin) / (1.0 + (shift - 1.0) * t_lin)
        return t.clamp(0.0, 1.0 - 1e-6)

    for i in range(steps, 0, -1):
        t_lin = torch.full((batch_size,), i / steps, device=device, dtype=torch.float32)
        t_next_lin = torch.full((batch_size,), (i - 1) / steps, device=device, dtype=torch.float32)

        t = flow_shift(t_lin)
        t_next = flow_shift(t_next_lin)

        if not use_cfg:
            x0_hat = model(x, t, y=y)
        else:
            y_cond = y
            y_uncond = torch.full_like(y, null_class)
            x0_hat_cond = model(x, t, y=y_cond)
            x0_hat_uncond = model(x, t, y=y_uncond)
            x0_hat = x0_hat_uncond + cfg_scale * (x0_hat_cond - x0_hat_uncond)

        t_scalar = t.view(batch_size, 1, 1, 1)
        eps_hat = (x - (1.0 - t_scalar) * x0_hat) / (t_scalar + 1e-8)

        t_next_scalar = t_next.view(batch_size, 1, 1, 1)
        x = t_next_scalar * eps_hat + (1.0 - t_next_scalar) * x0_hat

    return x


def build_dc_sae(sae_cfg, device: torch.device) -> DCSAE:
    """Build DCSAE model from config."""
    print("sae_cfg:", sae_cfg)
    return DCSAE(
        encoder_type=sae_cfg.encoder.type,
        dinov3_model_dir=sae_cfg.encoder.dinov3_model_dir,
        siglip2_model_name=sae_cfg.encoder.siglip2_model_name,
        dinov2_model_name=sae_cfg.encoder.dinov2_model_name,
        qwen3_vit_model_name=getattr(sae_cfg.encoder, "qwen3_vit_model_name", "Qwen/Qwen3-VL-8B-Instruct"),
        internvl3_model_name=getattr(sae_cfg.encoder, "internvl3_model_name", "OpenGVLab/InternVL3-8B"),
        internvl3_resize_target=getattr(sae_cfg.encoder, "internvl3_resize_target", 224),
        image_size=sae_cfg.data.image_size,
        in_channels=3,
        out_channels=3,
        hidden_size=sae_cfg.model.hidden_size,
        hidden_size_x=sae_cfg.model.hidden_size_x,
        decoder_type=getattr(sae_cfg.model, "decoder_type", "vit_decoder"),
        num_decoder_blocks=sae_cfg.model.num_decoder_blocks,
        nerf_max_freqs=sae_cfg.model.nerf_max_freqs,
        flow_steps=sae_cfg.model.flow_steps,
        time_dim=sae_cfg.model.time_dim,
        vit_decoder_hidden_size=getattr(sae_cfg.model, "vit_decoder_hidden_size", 1024),
        vit_decoder_num_layers=getattr(sae_cfg.model, "vit_decoder_num_layers", 24),
        vit_decoder_num_heads=getattr(sae_cfg.model, "vit_decoder_num_heads", 16),
        vit_decoder_intermediate_size=getattr(sae_cfg.model, "vit_decoder_intermediate_size", 4096),
        vit_decoder_dropout=getattr(sae_cfg.model, "vit_decoder_dropout", 0.0),
        gradient_checkpointing=getattr(sae_cfg.model, "gradient_checkpointing", False),
        enable_hf_branch=sae_cfg.model.enable_hf_branch,
        hf_dim=sae_cfg.model.hf_dim,
        hf_encoder_type=getattr(sae_cfg.model, "hf_encoder_type", "cnn"),
        hf_encoder_config_path=getattr(sae_cfg.model, "hf_encoder_config_path", None),
        hf_token_norm=getattr(sae_cfg.model, "hf_token_norm", False),
        hf_dropout_prob=0.0,  # Disable dropout for encoding
        hf_noise_std=0.0,
        hf_noise_alpha_schedule=getattr(sae_cfg.model, "hf_noise_alpha_schedule", "alpha_one"),
        hf_loss_weight=sae_cfg.model.hf_loss_weight,
        recon_l2_weight=sae_cfg.loss.recon_l2_weight,
        recon_l1_weight=sae_cfg.loss.recon_l1_weight,
        recon_lpips_weight=sae_cfg.loss.recon_lpips_weight,
        recon_gan_weight=sae_cfg.loss.recon_gan_weight,
        lora_rank=sae_cfg.model.lora_rank,
        lora_alpha=sae_cfg.model.lora_alpha,
        lora_dropout=sae_cfg.model.lora_dropout,
        enable_lora=sae_cfg.model.enable_lora,
        target_latent_channels=sae_cfg.model.target_latent_channels,
        variational=sae_cfg.model.variational,
        kl_weight=sae_cfg.model.kl_weight,
        skip_to_moments=sae_cfg.model.skip_to_moments,
        noise_tau=0.0,
        random_masking_channel_ratio=0.0,
        denormalize_decoder_output=sae_cfg.model.denormalize_decoder_output,
        enable_de_merger=getattr(sae_cfg.model, "enable_de_merger", False),
        de_merger_expand_size=getattr(sae_cfg.model, "de_merger_expand_size", 2),
        de_merger_num_layers=getattr(sae_cfg.model, "de_merger_num_layers", 2),
        de_merger_hidden_dim=getattr(sae_cfg.model, "de_merger_hidden_dim", None),
        de_merger_nhead=getattr(sae_cfg.model, "de_merger_nhead", 8),
    ).to(device)


def load_sae_checkpoint(model: DCSAE, ckpt_path: str, device: torch.device, logger):
    """Load dc-sae checkpoint."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"SAE checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in payload:
        state_dict = payload["model"]
    elif "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    step = payload.get("step", 0) if isinstance(payload, dict) else 0
    logger.info(f"Loaded SAE checkpoint from {ckpt_path} (step={step})")
    logger.info(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    return step


@torch.no_grad()
def encode_with_dc_sae(
    sae_model: DCSAE,
    x_img: torch.Tensor,
    zero_hf: bool = False,
) -> torch.Tensor:
    """
    Encode images using dc-sae.
    Returns fused latent [B, N, semantic_channels + hf_dim] reshaped to [B, C, H, W].
    
    Args:
        sae_model: dc-sae model
        x_img: Input images [B, 3, H, W]
        zero_hf: If True, zero out the HF channels
    """
    sae_model.eval()
    
    # Get semantic latent
    enc = sae_model._infer_latent(x_img)
    z = enc.latent  # [B, semantic_channels, H_p, W_p]
    
    # Get fused encoding (semantic + HF)
    # force_drop_hf=True will zero out HF tokens
    enc_cond = sae_model.encode(z, x_img=x_img, force_drop_hf=zero_hf)  # [B, N, C]
    
    # Reshape to spatial format [B, C, H, W]
    B, N, C = enc_cond.shape
    H = W = int(math.sqrt(N))
    latent = enc_cond.transpose(1, 2).contiguous().view(B, C, H, W)
    
    return latent


@torch.no_grad()
def decode_with_dc_sae(
    sae_model: DCSAE,
    latent: torch.Tensor,
    zero_hf: bool = False,
    latent_normalizer=None,
) -> torch.Tensor:
    """
    Decode latent using dc-sae.
    latent: [B, C, H, W] fused latent
    Returns: [B, 3, image_size, image_size] reconstructed images
    
    Args:
        sae_model: dc-sae model
        latent: Fused latent [B, C, H, W] where C = semantic_channels + hf_dim
        zero_hf: If True, zero out the HF channels before decoding
    """
    sae_model.eval()

    if latent_normalizer is not None:
        latent = latent_normalizer.denormalize(latent)
    
    B, C, H, W = latent.shape
    
    # Zero out HF channels if requested
    if zero_hf and sae_model.hf_dim > 0:
        latent = latent.clone()
        semantic_channels = sae_model.semantic_channels
        latent[:, semantic_channels:, :, :] = 0.0
    
    # Reshape to [B, N, C]
    enc_cond = latent.view(B, C, H * W).transpose(1, 2).contiguous()
    
    # Generate image
    img = sae_model.generate(enc_cond)
    return img.clamp(-1.0, 1.0)


#################################################################################
#                              FID Evaluation                                   #
#################################################################################

def create_flat_temp_dir(source_root, temp_dir):
    """Flatten the directory structure for pytorch-fid."""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    count = 0
    for root, dirs, files in os.walk(source_root):
        if os.path.abspath(root) == os.path.abspath(temp_dir):
            continue
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.abspath(os.path.join(root, file))
                parent_name = os.path.basename(root)
                new_name = f"{parent_name}_{file}"
                dst_path = os.path.join(temp_dir, new_name)
                try:
                    os.symlink(src_path, dst_path)
                    count += 1
                except OSError:
                    shutil.copy(src_path, dst_path)
                    count += 1
    return count


def _remove_file_if_exists(path: str):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def cleanup_flat_temp_dir_distributed(flat_dir: str, rank: int, world_size: int) -> int:
    """Delete flattened eval samples cooperatively across all ranks."""
    if not os.path.isdir(flat_dir):
        return 0

    entries = sorted(os.listdir(flat_dir))
    my_entries = entries[rank::world_size]
    my_paths = [os.path.join(flat_dir, name) for name in my_entries]
    if not my_paths:
        return 0

    with ThreadPoolExecutor(max_workers=min(128, len(my_paths))) as executor:
        list(executor.map(_remove_file_if_exists, my_paths))
    return len(my_paths)


def _remove_dir_if_exists(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def cleanup_eval_class_dirs_distributed(eval_dir: str, class_ids: list[int]) -> int:
    """Each rank deletes the class directories it created."""
    class_dirs = [os.path.join(eval_dir, str(cls_id)) for cls_id in class_ids]
    existing_dirs = [path for path in class_dirs if os.path.isdir(path)]
    if not existing_dirs:
        return 0

    with ThreadPoolExecutor(max_workers=min(32, len(existing_dirs))) as executor:
        list(executor.map(_remove_dir_if_exists, existing_dirs))
    return len(existing_dirs)


@torch.no_grad()
def run_fid_evaluation(
    model,
    sae_model: DCSAE,
    results_dir: str,
    fid_ref_path: str,
    fid_samples_per_class: int,
    fid_batch_size: int,
    latent_shape: tuple,
    time_shift: float,
    device: torch.device,
    step: int,
    rank: int,
    world_size: int,
    logger,
    writer=None,
    wandb_run=None,
    num_classes: int = 1000,
    ref_images_path: str = None,
    zero_hf: bool = False,
    latent_normalizer=None,
):
    """
    Run FID, IS, and Recall evaluation by generating samples and computing metrics.
    
    Args:
        model: DiT model (EMA)
        sae_model: dc-sae model for decoding
        results_dir: Directory to save generated samples
        fid_ref_path: Path to reference FID statistics (.npz)
        fid_samples_per_class: Number of samples per class
        fid_batch_size: Batch size for generation
        latent_shape: Shape of latent (C, H, W)
        time_shift: Time shift for flow matching
        device: Device to use
        step: Current training step
        rank: DDP rank
        world_size: DDP world size
        logger: Logger instance
        writer: TensorBoard writer
        num_classes: Number of classes (default 1000 for ImageNet)
        ref_images_path: Path to reference images directory for Recall calculation
        zero_hf: If True, zero out HF channels during decoding
    """
    if not HAS_FID and not HAS_TORCH_FIDELITY:
        if rank == 0:
            logger.warning("All metrics skipped because no evaluation library is installed.")
        return

    # Directories
    eval_dir = os.path.join(results_dir, "eval_samples", f"step_{step:07d}")
    flat_dir = os.path.join(results_dir, "eval_samples", f"step_{step:07d}_flat")

    if rank == 0:
        logger.info(f"========== Starting Evaluation (Step {step}) ==========")
        os.makedirs(eval_dir, exist_ok=True)

    dist.barrier()

    # Distribute classes across ranks
    all_classes = list(range(num_classes))
    my_classes = all_classes[rank::world_size]

    # Generation loop
    model.eval()

    total_needed = fid_samples_per_class

    if rank == 0:
        logger.info(f"Generating {total_needed} samples per class for {num_classes} classes...")

    for cls_id in my_classes:
        class_dir = os.path.join(eval_dir, str(cls_id))
        os.makedirs(class_dir, exist_ok=True)

        generated_count = 0
        while generated_count < total_needed:
            curr_batch = min(fid_batch_size, total_needed - generated_count)

            y = torch.full((curr_batch,), cls_id, device=device, dtype=torch.long)

            # FP32 for evaluation precision
            with torch.autocast(device_type='cuda', enabled=False):
                # 1. Sample latent (50 steps, no CFG)
                z_sample = sample_latent(
                    model=model,
                    batch_size=curr_batch,
                    latent_shape=latent_shape,
                    device=device,
                    y=y,
                    time_shift=time_shift,
                    steps=50,
                    use_cfg=False,
                )

                # 2. Decode with dc-sae (force drop HF when zero_hf is True)
                imgs = decode_with_dc_sae(
                    sae_model,
                    z_sample.float(),
                    zero_hf=zero_hf,
                    latent_normalizer=latent_normalizer,
                )

            # Convert to [0, 1] range
            imgs = (imgs + 1.0) / 2.0
            imgs = imgs.clamp(0, 1)

            # Save images
            for i in range(curr_batch):
                img_idx = generated_count + i
                save_image(imgs[i], os.path.join(class_dir, f"{img_idx}.png"))

            generated_count += curr_batch

    dist.barrier()

    # Metrics calculation (rank 0 only)
    if rank == 0:
        logger.info("Generation finished. Flattening directory for evaluation...")
        
        fid_value = None
        is_value = None
        precision_value = None
        recall_value = None
        
        try:
            num_imgs = create_flat_temp_dir(eval_dir, flat_dir)
            if num_imgs < 100:
                logger.warning(f"Too few images ({num_imgs}) for valid metrics.")
            else:
                # ============ FID Calculation (pytorch-fid) ============
                if HAS_FID and fid_ref_path is not None and os.path.exists(fid_ref_path):
                    logger.info(f"Calculating FID using {num_imgs} images...")
                    try:
                        fid_value = fid_score.calculate_fid_given_paths(
                            paths=[flat_dir, fid_ref_path],
                            batch_size=50,
                            device=device,
                            dims=2048,
                            num_workers=8
                        )
                        logger.info(f"Step {step} FID: {fid_value:.4f}")
                    except Exception as e:
                        logger.error(f"FID Calculation failed: {e}")
                
                # ============ IS, Precision, Recall (torch-fidelity) ============
                if HAS_TORCH_FIDELITY:
                    logger.info(f"Calculating IS, Precision, Recall using {num_imgs} images...")
                    try:
                        # Calculate Inception Score (doesn't need reference)
                        is_metrics = calculate_metrics(
                            input1=flat_dir,
                            cuda=True,
                            isc=True,  # Inception Score
                            isc_splits=10,
                            verbose=False,
                        )
                        is_value = is_metrics.get('inception_score_mean', None)
                        is_std = is_metrics.get('inception_score_std', None)
                        if is_value is not None:
                            logger.info(f"Step {step} IS: {is_value:.4f} ± {is_std:.4f}")
                        
                        # Calculate Precision and Recall (needs reference images)
                        if ref_images_path is not None and os.path.exists(ref_images_path):
                            logger.info(f"Calculating Precision/Recall with reference: {ref_images_path}")
                            pr_metrics = calculate_metrics(
                                input1=flat_dir,
                                input2=ref_images_path,
                                cuda=True,
                                prc=True,  # Precision and Recall
                                verbose=False,
                            )
                            precision_value = pr_metrics.get('precision', None)
                            recall_value = pr_metrics.get('recall', None)
                            if precision_value is not None:
                                logger.info(f"Step {step} Precision: {precision_value:.4f}")
                            if recall_value is not None:
                                logger.info(f"Step {step} Recall: {recall_value:.4f}")
                        else:
                            logger.warning("Recall/Precision skipped: ref_images_path not provided or invalid")
                            
                    except Exception as e:
                        logger.error(f"torch-fidelity metrics calculation failed: {e}")
                
                # ============ Log to TensorBoard ============
                if writer is not None:
                    if fid_value is not None:
                        writer.add_scalar("eval/FID", fid_value, global_step=step)
                    if is_value is not None:
                        writer.add_scalar("eval/IS", is_value, global_step=step)
                    if precision_value is not None:
                        writer.add_scalar("eval/Precision", precision_value, global_step=step)
                    if recall_value is not None:
                        writer.add_scalar("eval/Recall", recall_value, global_step=step)

                # ============ Log to Wandb ============
                if wandb_run is not None:
                    eval_log = {}
                    if fid_value is not None:
                        eval_log["eval/FID"] = fid_value
                    if is_value is not None:
                        eval_log["eval/IS"] = is_value
                    if precision_value is not None:
                        eval_log["eval/Precision"] = precision_value
                    if recall_value is not None:
                        eval_log["eval/Recall"] = recall_value
                    if eval_log:
                        wandb_run.log(eval_log, step=step)

                # ============ Write to text file ============
                with open(os.path.join(results_dir, "eval_metrics.txt"), "a") as f:
                    f.write(f"Step {step}:\n")
                    if fid_value is not None:
                        f.write(f"  FID: {fid_value:.4f}\n")
                    if is_value is not None:
                        f.write(f"  IS: {is_value:.4f}\n")
                    if precision_value is not None:
                        f.write(f"  Precision: {precision_value:.4f}\n")
                    if recall_value is not None:
                        f.write(f"  Recall: {recall_value:.4f}\n")
                    f.write("\n")

        except Exception as e:
            logger.error(f"Metrics Calculation failed: {e}")

    # Rank 0 does metrics only; cleanup is distributed afterward to avoid
    # leaving other ranks blocked in a collective while rank 0 deletes files.
    dist.barrier()

    if rank == 0:
        logger.info("Starting distributed cleanup of evaluation samples...")

    cleanup_flat_temp_dir_distributed(flat_dir, rank, world_size)
    dist.barrier()

    if rank == 0:
        if os.path.exists(flat_dir) and not os.listdir(flat_dir):
            os.rmdir(flat_dir)

    cleanup_eval_class_dirs_distributed(eval_dir, my_classes)
    dist.barrier()

    # Rank 0 removes the empty eval_dir after all ranks finish cleanup
    if rank == 0:
        if os.path.exists(eval_dir) and not os.listdir(eval_dir):
            os.rmdir(eval_dir)
    
    logger.info(f"========== End Evaluation (Step {step}) ==========")


#################################################################################
#                                  Main Training                                #
#################################################################################

def main():
    parser = argparse.ArgumentParser(description="Train DiT on dc-sae latent space")
    parser.add_argument("--config", type=str, required=True, help="DiT config path")
    parser.add_argument("--sae-config", type=str, required=True, help="dc-sae config path")
    parser.add_argument("--sae-ckpt", type=str, required=True, help="dc-sae checkpoint path")
    parser.add_argument("--data-path", type=str, default=None, help="ImageNet train path (required for imagefolder mode)")
    parser.add_argument("--results-dir", type=str, default="results_dit", help="Output directory")
    parser.add_argument("--ckpt", type=str, default=None, help="Resume checkpoint")
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="If --ckpt is unset, automatically resume from results_dir/checkpoints.",
    )
    parser.add_argument("--precision", type=str, choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--cfg-prob", type=float, default=0.1, help="CFG dropout probability")
    parser.add_argument("--cfg-scale", type=float, default=3.0, help="CFG scale for sampling")
    
    # Evaluation args (FID, IS, Recall)
    parser.add_argument("--fid-ref-path", type=str, default="VIRTUAL_imagenet256_labeled.npz",
                        help="Path to reference FID statistics (.npz)")
    parser.add_argument("--ref-images-path", type=str, default=None,
                        help="Path to reference images directory for Precision/Recall calculation")
    parser.add_argument("--fid-samples-per-class", type=int, default=50,
                        help="Number of samples per class for FID evaluation")
    parser.add_argument("--fid-batch-size", type=int, default=32,
                        help="Batch size for FID generation")
    parser.add_argument("--skip-fid", action="store_true",
                        help="Skip FID/IS/Recall evaluation")
    parser.add_argument(
        "--latent-stats-path",
        type=str,
        default=None,
        help="Path to latent stats file (.pt/.npz) for latent normalization. If not set, read from config misc.latent_stats_path.",
    )
    parser.add_argument(
        "--per-channel-norm",
        action="store_true",
        help="Use per-channel normalization when --latent-stats-path is provided.",
    )
    parser.add_argument(
        "--use-env-paths",
        action="store_true",
        help="Prefer known path environment variables over CLI path values.",
    )
    # WDS dataloader args
    parser.add_argument(
        "--data-mode", type=str, default="imagefolder",
        choices=["imagefolder", "wds_offline", "wds_online"],
        help="Data loading mode: imagefolder (default), wds_offline, or wds_online",
    )
    parser.add_argument("--wds-root", type=str, default=None, help="WDS root directory")
    parser.add_argument("--feat-key", type=str, default=None, help="WDS feature key for offline mode")
    args = parser.parse_args()
    arg_env_overrides = apply_path_env_overrides(
        args,
        DIT_TRAIN_ARG_PATH_ENV_VARS,
        enabled=args.use_env_paths,
    )

    # Parse DiT config
    (
        _,  # rae_config (not used)
        model_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        training_config,
    ) = parse_configs(args.config)

    def to_dict(cfg_section):
        if cfg_section is None:
            return {}
        return OmegaConf.to_container(cfg_section, resolve=True)

    misc = to_dict(misc_config)
    training_cfg = to_dict(training_config)

    # Training hyperparameters from config
    num_classes = int(misc.get("num_classes", 1000))
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    clip_grad = float(training_cfg.get("clip_grad", 1.0))
    ema_decay = float(training_cfg.get("ema_decay", 0.9995))
    epochs = int(training_cfg.get("epochs", 1400))
    global_batch_size = int(training_cfg.get("global_batch_size", 512))
    num_workers = int(training_cfg.get("num_workers", 4))
    log_every = int(training_cfg.get("log_every", 100))
    balance_every = int(training_cfg.get("balance_every", 0))
    ckpt_every = int(training_cfg.get("ckpt_every", 10000))
    sample_every = int(training_cfg.get("sample_every", 10000))
    global_seed = int(training_cfg.get("global_seed", 0))

    # Latent size from config
    latent_size_cfg = misc.get("latent_size", [832, 16, 16])
    latent_size = tuple(latent_size_cfg)
    latent_channels = latent_size[0]

    # Time shift
    shift_dim = latent_size[0] * latent_size[1] * latent_size[2]
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_shift = math.sqrt(shift_dim / shift_base)

    # Zero HF mode: when True, HF channels are zeroed and excluded from loss
    zero_hf_mode = bool(misc.get("zero_hf_mode", False))

    # hf_zero_then_joint mode: train with zero HF first, then switch to joint training
    hf_zero_then_joint_mode = bool(misc.get("hf_zero_then_joint_mode", False))
    joint_start_step = int(misc.get("joint_start_step", 0))
    if hf_zero_then_joint_mode and zero_hf_mode:
        zero_hf_mode = False  # hf_zero_then_joint supersedes pure zero_hf

    # DDP setup
    rank, local_rank, world_size = setup_ddp()
    device = torch.device("cuda", local_rank)

    if global_batch_size % (world_size * grad_accum_steps) != 0:
        raise ValueError("Global batch size must be divisible by world_size * grad_accum_steps.")
    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)

    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    distributed_mode = str(training_cfg.get("distributed", "ddp")).lower()
    if distributed_mode not in {"ddp", "fsdp"}:
        raise ValueError(f"Unsupported distributed mode: {distributed_mode}")

    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}")
        print(f"Latent size: {latent_size}, time_shift: {time_shift:.4f}")
        print(f"Zero HF mode: {zero_hf_mode}")
        print(f"Distributed mode: {distributed_mode}")
        for item in arg_env_overrides:
            print(f"[path_env] args.{item['field']} <= ${item['env']} -> {item['value']}")
        if hf_zero_then_joint_mode:
            print(f"hf_zero_then_joint mode: ON, joint_start_step={joint_start_step}")
        if balance_every > 0:
            print(f"Gradient balance: ON, balance_every={balance_every}")

    # Create directories
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_dir = os.path.join(args.results_dir, "experiment")
        checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
        sample_dir = os.path.join(args.results_dir, "samples")
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)

        logger = create_logger(experiment_dir, rank=0)
        logger.info(f"Experiment directory: {experiment_dir}")

        tb_dir = os.path.join(experiment_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir) if HAS_TENSORBOARD else None
    else:
        experiment_dir = None
        checkpoint_dir = None
        sample_dir = None
        logger = create_logger(None, rank=rank)
        writer = None

    # Wandb setup
    wandb_run = None
    wandb_project = misc.get("wandb_project", "")
    if rank == 0 and HAS_WANDB and wandb_project:
        wandb_name = misc.get("wandb_name", "") or os.path.basename(args.results_dir)
        wandb_entity = misc.get("wandb_entity", "") or None
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            config={
                "dit_config": args.config,
                "sae_config": args.sae_config,
                "sae_ckpt": args.sae_ckpt,
                "precision": args.precision,
                "cfg_prob": args.cfg_prob,
                "cfg_scale": args.cfg_scale,
                "misc": misc,
                "training": training_cfg,
            },
            dir=args.results_dir,
            resume="allow",
        )
        logger.info(f"Wandb logging to project={wandb_project}, run={wandb_name}")

    # Load dc-sae
    logger.info(f"Loading dc-sae from {args.sae_ckpt}...")
    sae_cfg = load_config(args.sae_config, use_env_paths=args.use_env_paths)
    sae_config_env_overrides = apply_path_env_overrides(
        sae_cfg,
        SAE_CONFIG_PATH_ENV_VARS,
        enabled=args.use_env_paths,
    )
    if rank == 0:
        for item in sae_config_env_overrides:
            logger.info(f"[path_env] sae_config.{item['field']} <= ${item['env']} -> {item['value']}")
    sae_model = build_dc_sae(sae_cfg, device)
    load_sae_checkpoint(sae_model, args.sae_ckpt, device, logger)
    sae_model.eval()
    requires_grad(sae_model, False)
    logger.info(f"dc-sae loaded. Semantic channels: {sae_model.semantic_channels}, HF dim: {sae_model.hf_dim}")

    latent_stats_path = args.latent_stats_path if args.latent_stats_path is not None else misc.get("latent_stats_path", None)
    per_channel_norm = bool(misc.get("per_channel_norm", False)) or bool(args.per_channel_norm)

    latent_normalizer = None
    if latent_stats_path:
        latent_stats = load_latent_stats(latent_stats_path, device=device, verbose=(rank == 0))
        latent_normalizer = LatentNormalizer(latent_stats, per_channel=per_channel_norm)
        if rank == 0:
            norm_mode = "per-channel" if per_channel_norm else "scalar"
            logger.info(f"Latent normalization: {norm_mode}, stats={latent_stats_path}")
    elif rank == 0:
        logger.info("Latent normalization: disabled")

    time_shift_full = time_shift
    time_shift_semantic = math.sqrt((sae_model.semantic_channels * latent_size[1] * latent_size[2]) / shift_base)

    if zero_hf_mode:
        time_shift = time_shift_semantic
        if rank == 0:
            logger.info("Zero HF mode is enabled. HF channels are zeroed during training and evaluation.")
            logger.info(f"Semantic channels: {sae_model.semantic_channels}, HF dim: {sae_model.hf_dim}")
            logger.info(f"Time shift: {time_shift:.4f}")

    if hf_zero_then_joint_mode:
        time_shift = time_shift_semantic
        time_shift_full = time_shift_semantic  # keep time_shift unchanged across phases
        if rank == 0:
            logger.info(f"hf_zero_then_joint mode: joint training starts at step {joint_start_step}")
            logger.info(f"  Semantic channels: {sae_model.semantic_channels}, HF dim: {sae_model.hf_dim}")
            logger.info(f"  time_shift: {time_shift_semantic:.4f} (unchanged across phases)")

    # Verify latent dimensions match
    expected_channels = sae_model.semantic_channels + sae_model.hf_dim
    if expected_channels != latent_channels:
        raise ValueError(
            f"Latent channel mismatch: SAE outputs {expected_channels} channels, "
            f"but DiT config expects {latent_channels}"
        )

    # Build DiT model
    model = instantiate_from_config(model_config).to(device)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    opt_state = None
    sched_state = None
    train_steps = 0
    ema_loaded = False
    resume_ckpt = args.ckpt

    if (not resume_ckpt) and args.auto_resume:
        resume_ckpt, auto_resume_note = find_auto_resume_checkpoint(args.results_dir)
        if rank == 0:
            if resume_ckpt:
                logger.info(f"Auto-resume enabled. {auto_resume_note} Resume checkpoint: {resume_ckpt}")
            else:
                logger.info(f"Auto-resume enabled, but no checkpoint was found. {auto_resume_note}")

    # Resume checkpoint (skip if None, empty string, or file doesn't exist)
    if resume_ckpt and os.path.isfile(resume_ckpt):
        checkpoint = torch.load(resume_ckpt, map_location="cpu")
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        if "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"], strict=False)
            ema_loaded = True
        opt_state = checkpoint.get("opt")
        sched_state = checkpoint.get("scheduler")
        train_steps = int(checkpoint.get("train_steps", 0))
        if rank == 0:
            logger.info(f"Resumed from {resume_ckpt}, train_steps={train_steps}")
    else:
        if rank == 0:
            logger.info("No checkpoint provided or file not found, training from scratch.")

    model_param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"DiT Parameters: {model_param_count / 1e6:.2f}M")

    # DDP/FSDP wrap
    model = build_wrapped_dit_model(model, training_cfg, local_rank, args.precision)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Optimizer
    opt, opt_msg = build_optimizer(model.parameters(), training_cfg)
    if opt_state is not None:
        opt.load_state_dict(opt_state)

    # Dataset
    wds_dataset = None  # set when using WDS modes
    if args.data_mode in ("wds_offline", "wds_online"):
        from data.imagenet_wds import ImageNetWDS
        wds_mode = "offline" if args.data_mode == "wds_offline" else "online"
        encoder_for_wds = None
        if wds_mode == "online":
            from data.encoders import DCSAEEncoder
            encoder_for_wds = DCSAEEncoder(
                sae_config_path=args.sae_config,
                sae_ckpt_path=args.sae_ckpt,
                latent_stats_path=args.latent_stats_path,
                image_size=sae_cfg.data.image_size,
                device=device,
                precision=args.precision,
            )
        wds_dataset = ImageNetWDS(
            wds_root=args.wds_root,
            mode=wds_mode,
            feat_key=args.feat_key,
            encoder=encoder_for_wds,
            image_size=sae_cfg.data.image_size,
            shuffle=True,
            seed=global_seed,
        )
        loader = DataLoader(
            wds_dataset,
            batch_size=micro_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        sampler = None
        logger.info(f"Dataset: WDS {wds_mode} from {args.wds_root}")
    else:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, sae_cfg.data.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t * 2.0 - 1.0),
        ])
        dataset = ImageFolder(args.data_path, transform=transform)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=global_seed)
        loader = DataLoader(
            dataset,
            batch_size=micro_batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        logger.info(f"Dataset: {len(dataset):,} images from {args.data_path}")
    logger.info(f"Batch: micro={micro_batch_size}, global={global_batch_size}, accum={grad_accum_steps}")
    logger.info(f"Precision: {args.precision}")

    try:
        loader_batches = len(loader)
    except TypeError:
        # IterableDataset doesn't support len() well; estimate from wdinfo
        if wds_dataset is not None:
            loader_batches = max(1, len(wds_dataset) // (micro_batch_size * world_size))
        else:
            loader_batches = 1
    steps_per_epoch = max(1, loader_batches // grad_accum_steps)

    schedl, sched_msg = build_scheduler(opt, steps_per_epoch, training_cfg, sched_state)
    if rank == 0:
        logger.info(f"Training: {epochs} epochs, {steps_per_epoch} steps/epoch")
        logger.info(opt_msg + "\n" + sched_msg)

    # Initialize EMA from model only when checkpoint EMA is unavailable.
    if not ema_loaded:
        update_ema(ema, model, decay=0, use_fsdp=(distributed_mode == "fsdp"))
    model.train()
    ema.eval()

    # AMP
    use_amp = (args.precision == "bf16")
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    log_steps = 0
    running_loss = 0.0
    running_sem_loss = 0.0
    running_hf_loss = 0.0
    start_time = time()
    joint_phase_started = (train_steps >= joint_start_step) if hf_zero_then_joint_mode else False
    last_sem_grad_norm = None
    last_hf_grad_norm = None
    sem_loss_weight = 1.0
    hf_loss_weight = 1.0
    grad_ratio_ema = 1.0

    logger.info("Starting training...")

    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        if wds_dataset is not None:
            wds_dataset.set_epoch(epoch)
        if rank == 0:
            print(f"Beginning epoch {epoch}...")

        opt.zero_grad()
        accum_counter = 0
        step_loss_accum = 0.0
        step_sem_loss_accum = 0.0
        step_hf_loss_accum = 0.0

        for data_batch in loader:
            if args.data_mode == "wds_offline":
                x_latent, y = data_batch
                x_latent = x_latent.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
            elif args.data_mode == "wds_online":
                x_latent, y = data_batch
                x_latent = x_latent.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
            else:
                x_img, y = data_batch
                x_img = x_img.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

            # Determine current phase
            if hf_zero_then_joint_mode:
                in_joint_phase = (train_steps >= joint_start_step)
                current_zero_hf = not in_joint_phase
                current_time_shift = time_shift_full if in_joint_phase else time_shift_semantic
                component_ch = sae_model.semantic_channels if in_joint_phase else None
            else:
                in_joint_phase = False
                current_zero_hf = zero_hf_mode
                current_time_shift = time_shift
                component_ch = None

            # Encode with dc-sae (skip for WDS modes - already encoded)
            if args.data_mode == "imagefolder":
                with torch.no_grad(), autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    x_latent = encode_with_dc_sae(sae_model, x_img, zero_hf=False)
                    if latent_normalizer is not None:
                        x_latent = latent_normalizer.normalize(x_latent)
            # CFG label dropout
            NULL_CLASS = num_classes
            if args.cfg_prob > 0.0:
                drop_mask = torch.rand_like(y.float()) < args.cfg_prob
                y_train = y.clone()
                y_train[drop_mask] = NULL_CLASS
            else:
                y_train = y

            model_kwargs = dict(y=y_train)

            # Forward + loss
            semantic_channels_for_loss = sae_model.semantic_channels if current_zero_hf else None
            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                loss_tensor, pred_latent, noise, t_sample, loss_components = compute_train_loss(
                    model=model,
                    x_latent=x_latent,
                    model_kwargs=model_kwargs,
                    time_shift=time_shift,
                    semantic_channels=semantic_channels_for_loss,
                    component_split_ch=component_ch,
                )

            # NaN check
            is_nan_local = 0 if torch.isfinite(loss_tensor) else 1
            is_nan = torch.tensor(is_nan_local, device=device)
            dist.all_reduce(is_nan, op=dist.ReduceOp.SUM)

            if is_nan.item() > 0:
                if rank == 0:
                    logger.warning(f"[step {train_steps}] NaN detected! Skipping...")
                opt.zero_grad(set_to_none=True)
                accum_counter = 0
                step_loss_accum = 0.0
                step_sem_loss_accum = 0.0
                step_hf_loss_accum = 0.0
                dist.barrier()
                continue

            # Track component losses
            if loss_components:
                step_sem_loss_accum += loss_components["semantic_loss"].item()
                step_hf_loss_accum += loss_components["hf_loss"].item()

            # Dynamic loss reweighting for semantic/HF gradient balance in joint phase.
            if in_joint_phase and loss_components and balance_every > 0:
                loss_tensor = (
                    sem_loss_weight * loss_components["semantic_loss"]
                    + hf_loss_weight * loss_components["hf_loss"]
                )

            # Compute separate grad norms:
            # - for logging at log boundary
            # - for dynamic reweighting at balance boundary
            need_balance_grads = (
                in_joint_phase
                and loss_components
                and balance_every > 0
                and accum_counter == grad_accum_steps - 1
                and (train_steps + 1) % balance_every == 0
            )
            need_component_grads = (
                in_joint_phase
                and loss_components
                and accum_counter == grad_accum_steps - 1
                and (train_steps + 1) % log_every == 0
            )
            need_grad_stats = need_balance_grads or need_component_grads

            if need_grad_stats:
                sem_grads = torch.autograd.grad(
                    loss_components["semantic_loss"], trainable_params,
                    retain_graph=True, allow_unused=True,
                )
                last_sem_grad_norm = torch.sqrt(
                    sum((g.float().norm() ** 2) for g in sem_grads if g is not None)
                ).item()

                hf_grads = torch.autograd.grad(
                    loss_components["hf_loss"], trainable_params,
                    retain_graph=True, allow_unused=True,
                )
                last_hf_grad_norm = torch.sqrt(
                    sum((g.float().norm() ** 2) for g in hf_grads if g is not None)
                ).item()

                if need_balance_grads:
                    eps = 1e-8
                    ratio = (last_sem_grad_norm + eps) / (last_hf_grad_norm + eps)
                    grad_ratio_ema = 0.9 * grad_ratio_ema + 0.1 * ratio
                    adj = grad_ratio_ema ** 0.5
                    sem_loss_weight = 1.0 / max(adj, eps)
                    hf_loss_weight = max(adj, eps)
                    sem_loss_weight = float(min(max(sem_loss_weight, 0.2), 5.0))
                    hf_loss_weight = float(min(max(hf_loss_weight, 0.2), 5.0))
                    norm = max(0.5 * (sem_loss_weight + hf_loss_weight), eps)
                    sem_loss_weight /= norm
                    hf_loss_weight /= norm

            # Backward
            step_loss_accum += loss_tensor.item()
            loss_tensor = loss_tensor / grad_accum_steps
            loss_tensor.backward()
            accum_counter += 1

            if accum_counter < grad_accum_steps:
                continue

            # Optimizer step
            if clip_grad > 0:
                grad_norm = clip_grad_norm(model, clip_grad)
            else:
                grad_norm = torch.tensor(0.0, device=device)

            opt.step()
            schedl.step()
            update_ema(ema, model, decay=ema_decay, use_fsdp=(distributed_mode == "fsdp"))
            opt.zero_grad()

            running_loss += step_loss_accum / grad_accum_steps
            if in_joint_phase:
                running_sem_loss += step_sem_loss_accum / grad_accum_steps
                running_hf_loss += step_hf_loss_accum / grad_accum_steps
            log_steps += 1
            train_steps += 1
            accum_counter = 0
            step_loss_accum = 0.0
            step_sem_loss_accum = 0.0
            step_hf_loss_accum = 0.0

            # Log phase transition
            if hf_zero_then_joint_mode and not joint_phase_started and train_steps >= joint_start_step:
                joint_phase_started = True
                if rank == 0:
                    logger.info("=" * 60)
                    logger.info(f"Switching to JOINT training at step {train_steps}")
                    logger.info(f"Time shift: {time_shift_full:.4f} (unchanged)")
                    logger.info("=" * 60)

            # Logging
            if train_steps % log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / max(1, log_steps), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size

                if rank == 0:
                    msg = (
                        f"(step={train_steps:07d}) Loss: {avg_loss:.4f}, "
                        f"Steps/Sec: {steps_per_sec:.2f}, Grad Norm: {float(grad_norm):.4f}"
                    )
                    if in_joint_phase and log_steps > 0:
                        avg_sem = running_sem_loss / log_steps
                        avg_hf = running_hf_loss / log_steps
                        msg += f", Sem Loss: {avg_sem:.4f}, HF Loss: {avg_hf:.4f}"
                        if last_sem_grad_norm is not None:
                            msg += f", Sem GradNorm: {last_sem_grad_norm:.4f}, HF GradNorm: {last_hf_grad_norm:.4f}"
                        if balance_every > 0:
                            msg += f", SemW: {sem_loss_weight:.3f}, HFW: {hf_loss_weight:.3f}"
                    print(msg)

                    if writer is not None:
                        writer.add_scalar("train/loss", avg_loss, global_step=train_steps)
                        writer.add_scalar("train/steps_per_sec", steps_per_sec, global_step=train_steps)
                        writer.add_scalar("train/grad_norm", float(grad_norm), global_step=train_steps)
                        if in_joint_phase and log_steps > 0:
                            writer.add_scalar("train/semantic_loss", avg_sem, global_step=train_steps)
                            writer.add_scalar("train/hf_loss", avg_hf, global_step=train_steps)
                            if last_sem_grad_norm is not None:
                                writer.add_scalar("train/semantic_grad_norm", last_sem_grad_norm, global_step=train_steps)
                                writer.add_scalar("train/hf_grad_norm", last_hf_grad_norm, global_step=train_steps)
                            if balance_every > 0:
                                writer.add_scalar("train/semantic_loss_weight", sem_loss_weight, global_step=train_steps)
                                writer.add_scalar("train/hf_loss_weight", hf_loss_weight, global_step=train_steps)

                    if wandb_run is not None:
                        wandb_log = {
                            "train/loss": avg_loss,
                            "train/steps_per_sec": steps_per_sec,
                            "train/grad_norm": float(grad_norm),
                        }
                        if in_joint_phase and log_steps > 0:
                            wandb_log["train/semantic_loss"] = avg_sem
                            wandb_log["train/hf_loss"] = avg_hf
                            if last_sem_grad_norm is not None:
                                wandb_log["train/semantic_grad_norm"] = last_sem_grad_norm
                                wandb_log["train/hf_grad_norm"] = last_hf_grad_norm
                            if balance_every > 0:
                                wandb_log["train/semantic_loss_weight"] = sem_loss_weight
                                wandb_log["train/hf_loss_weight"] = hf_loss_weight
                        wandb_run.log(wandb_log, step=train_steps)

                running_loss = 0.0
                running_sem_loss = 0.0
                running_hf_loss = 0.0
                log_steps = 0
                start_time = time()

            # Sampling visualization
            if train_steps % sample_every == 0 and rank == 0:
                with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
                    y_sample = y[:4]
                    sample_batch_size = y_sample.shape[0]
                    z_sample = sample_latent(
                        model=ema,
                        batch_size=sample_batch_size,
                        latent_shape=latent_size,
                        device=device,
                        y=y_sample,
                        time_shift=current_time_shift,
                        steps=50,
                        use_cfg=False,
                    )

                    img_gen = decode_with_dc_sae(
                        sae_model,
                        z_sample.float(),
                        zero_hf=current_zero_hf,
                        latent_normalizer=latent_normalizer,
                    )
                    img_gen_01 = (img_gen + 1.0) / 2.0
                    save_image(img_gen_01, os.path.join(sample_dir, f"sample_step_{train_steps:07d}.png"), nrow=2)

                    x_gt_01 = (x_img[:4] + 1.0) / 2.0
                    save_image(x_gt_01, os.path.join(sample_dir, f"gt_step_{train_steps:07d}.png"), nrow=2)

                    recon = decode_with_dc_sae(
                        sae_model,
                        pred_latent[:4].float(),
                        zero_hf=current_zero_hf,
                        latent_normalizer=latent_normalizer,
                    )
                    recon_01 = (recon + 1.0) / 2.0
                    save_image(recon_01, os.path.join(sample_dir, f"recon_step_{train_steps:07d}.png"), nrow=2)

                    if wandb_run is not None:
                        wandb_run.log({
                            "vis/generated": wandb.Image(
                                os.path.join(sample_dir, f"sample_step_{train_steps:07d}.png"),
                                caption="Generated",
                            ),
                            "vis/ground_truth": wandb.Image(
                                os.path.join(sample_dir, f"gt_step_{train_steps:07d}.png"),
                                caption="Ground truth",
                            ),
                            "vis/reconstruction": wandb.Image(
                                os.path.join(sample_dir, f"recon_step_{train_steps:07d}.png"),
                                caption="Reconstruction",
                            ),
                        }, step=train_steps)

                logger.info(f"[step {train_steps}] Saved samples")

            # Checkpoint
            if train_steps % ckpt_every == 0 and train_steps > 0:
                model_state = get_model_state_dict(model)
                optimizer_state = get_optimizer_state_dict(model, opt)
                if rank == 0:
                    checkpoint = {
                        "model": model_state,
                        "ema": ema.state_dict(),
                        "opt": optimizer_state,
                        "scheduler": schedl.state_dict(),
                        "train_steps": train_steps,
                        "config": args.config,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint: {checkpoint_path}")
                dist.barrier()

                # FID evaluation after checkpoint
                if not args.skip_fid:
                    if rank == 0:
                        logger.info(f"Step {train_steps}: Starting FID evaluation...")
                    
                    torch.cuda.empty_cache()
                    
                    run_fid_evaluation(
                        model=ema,
                        sae_model=sae_model,
                        results_dir=args.results_dir,
                        fid_ref_path=args.fid_ref_path,
                        fid_samples_per_class=args.fid_samples_per_class,
                        fid_batch_size=args.fid_batch_size,
                        latent_shape=latent_size,
                        time_shift=current_time_shift,
                        device=device,
                        step=train_steps,
                        rank=rank,
                        world_size=world_size,
                        logger=logger,
                        writer=writer,
                        wandb_run=wandb_run,
                        num_classes=num_classes,
                        ref_images_path=args.ref_images_path,
                        zero_hf=current_zero_hf,
                        latent_normalizer=latent_normalizer,
                    )
                    
                    model.train()
                    torch.cuda.empty_cache()

            # Latest snapshot
            if train_steps % 1000 == 0 and train_steps > 0:
                model_state = get_model_state_dict(model)
                optimizer_state = get_optimizer_state_dict(model, opt)
                if rank == 0:
                    snapshot = {
                        "model": model_state,
                        "ema": ema.state_dict(),
                        "opt": optimizer_state,
                        "scheduler": schedl.state_dict(),
                        "train_steps": train_steps,
                        "config": args.config,
                    }
                    latest_path = f"{checkpoint_dir}/latest.pt"
                    torch.save(snapshot, latest_path)
                    logger.info(f"Updated latest checkpoint: {latest_path}")
                dist.barrier()

    model.eval()
    if wandb_run is not None:
        wandb_run.finish()
    logger.info("Training complete!")
    cleanup_ddp()


if __name__ == "__main__":
    main()
