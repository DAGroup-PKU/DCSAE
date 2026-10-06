import math
import os
import random
import shutil
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

from config_utils import load_and_merge_config
from model import DCSAE
from models.rae.utils.ddp_utils import (
    FSDP,
    HAS_FSDP,
    cleanup_ddp,
    clip_grad_norm,
    create_logger,
    get_grad_norm,
    get_model_state_dict,
    get_optimizer_state_dict,
    requires_grad,
    setup_ddp,
    unwrap_model,
    wrap_model_ddp,
    wrap_model_fsdp,
)
from models.rae.utils.image_utils import center_crop_arr
from models.rae.utils.metrics_utils import calculate_batch_psnr, is_fid_available

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not found. TensorBoard logging will be skipped.")

HAS_FID = is_fid_available()
if HAS_FID:
    from pytorch_fid import fid_score
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Reuse GAN components from train_vae.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_VAE_DIR = os.path.join(ROOT_DIR, "train_vae")
if TRAIN_VAE_DIR not in sys.path:
    sys.path.insert(0, TRAIN_VAE_DIR)

from train_vae.dinodisc import DiffAug, DinoDisc, hinge_d_loss
from train_vae.gan_model import NLayerDiscriminator, d_hinge_loss


def _cfg_to_dict(cfg) -> dict:
    """Convert a nested dataclass config to a plain dict for wandb.config."""
    from dataclasses import fields, is_dataclass
    if is_dataclass(cfg):
        return {f.name: _cfg_to_dict(getattr(cfg, f.name)) for f in fields(cfg)}
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_wrapped_model(model: torch.nn.Module, cfg, local_rank: int):
    if cfg.training.distributed == "fsdp":
        ignored_modules = []
        for child in model.children():
            child_params = list(child.parameters())
            if child_params and all(not p.requires_grad for p in child_params):
                ignored_modules.append(child)

        auto_wrap_policy = None
        wrap_module_classes = set()
        decoder_layers = getattr(getattr(model, "decoder", None), "decoder_layers", None)
        if decoder_layers is not None and len(decoder_layers) > 0:
            wrap_module_classes.add(type(decoder_layers[0]))
        de_merger_transformer = getattr(getattr(getattr(model, "de_merger", None), "transformer", None), "layers", None)
        if de_merger_transformer is not None and len(de_merger_transformer) > 0:
            wrap_module_classes.add(type(de_merger_transformer[0]))
        encoder_merger_transformer = getattr(
            getattr(getattr(getattr(model, "encoder", None), "trainable_merger", None), "transformer", None),
            "layers",
            None,
        )
        if encoder_merger_transformer is not None and len(encoder_merger_transformer) > 0:
            wrap_module_classes.add(type(encoder_merger_transformer[0]))
        if wrap_module_classes:
            from torch.distributed.fsdp.wrap import ModuleWrapPolicy

            auto_wrap_policy = ModuleWrapPolicy(wrap_module_classes)

        return wrap_model_fsdp(
            model,
            device_id=local_rank,
            use_bf16=cfg.training.precision == "bf16",
            sharding_strategy=cfg.training.fsdp_sharding_strategy,
            use_orig_params=cfg.training.fsdp_use_orig_params,
            ignored_modules=ignored_modules or None,
            auto_wrap_policy=auto_wrap_policy,
        )
    return wrap_model_ddp(model, device_id=local_rank, find_unused_parameters=False)


def build_optimizer_for_model(model: torch.nn.Module, cfg):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.optimizer.lr,
        betas=(cfg.optimizer.betas0, cfg.optimizer.betas1),
        weight_decay=cfg.optimizer.weight_decay,
    )


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_scheduler(cfg, optimizer: torch.optim.Optimizer, current_step: int = 0):
    if cfg.optimizer.scheduler != "cosine":
        return None

    min_lr_ratio = cfg.optimizer.min_lr / cfg.optimizer.lr if cfg.optimizer.lr > 0 else 0.1
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        warmup_steps=cfg.optimizer.warmup_steps,
        total_steps=cfg.training.max_steps,
        min_lr_ratio=min_lr_ratio,
    )
    if current_step > 0:
        scheduler.last_epoch = current_step - 1
        scheduler.step()
    return scheduler


def _load_optimizer_state_dict(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    optimizer_state: Optional[dict],
    logger,
    name: str,
    rank: int,
) -> bool:
    if optimizer is None or optimizer_state is None:
        return False
    try:
        if HAS_FSDP and isinstance(model, FSDP):
            if rank == 0:
                full_state = optimizer_state
            else:
                full_state = None
            sharded_state = FSDP.scatter_full_optim_state_dict(
                full_state,
                model,
                optim=optimizer,
            )
            optimizer.load_state_dict(sharded_state)
        else:
            optimizer.load_state_dict(optimizer_state)
        return True
    except Exception as exc:
        if rank == 0:
            logger.warning(f"Skipping {name} state restore: {exc}")
        return False


class FlatImageDataset(Dataset):
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.paths = sorted(
            [
                os.path.join(root, f)
                for f in os.listdir(root)
                if os.path.splitext(f)[1].lower() in self.EXTS
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


def build_datasets(image_size: int, train_path: str, val_path: str):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t * 2.0 - 1.0),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: center_crop_arr(img, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t * 2.0 - 1.0),
        ]
    )

    train_dataset = ImageFolder(train_path, transform=train_transform)
    try:
        val_dataset = ImageFolder(val_path, transform=val_transform)
    except Exception:
        val_dataset = FlatImageDataset(val_path, transform=val_transform)
    return train_dataset, val_dataset


def reduce_mean_tensor(t: torch.Tensor) -> torch.Tensor:
    out = t.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out = out / dist.get_world_size()
    return out


@torch.no_grad()
def save_visualization(
    model: torch.nn.Module,
    vis_batch: torch.Tensor,
    step: int,
    output_dir: str,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
    rank: int,
    nrow: int = 4,
) -> Optional[str]:
    """Save visualization of ground truth and reconstructed images side by side."""
    model.eval()
    x = vis_batch.to(device, non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
        out = model(x)
    recon = out.sample.clamp(-1.0, 1.0)

    save_path = None
    if rank == 0:
        x_vis = (x + 1.0) / 2.0
        recon_vis = (recon + 1.0) / 2.0
        batch_size = x.shape[0]
        interleaved = torch.stack([x_vis, recon_vis], dim=1).view(batch_size * 2, *x_vis.shape[1:])
        grid = make_grid(interleaved, nrow=nrow * 2, padding=2, normalize=False)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        grid_np = (grid_np * 255).astype(np.uint8)
        img = Image.fromarray(grid_np)

        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        save_path = os.path.join(vis_dir, f"step_{step}.png")
        img.save(save_path)

    model.train()
    return save_path


def get_rfid_dirs(output_dir: str, step: int) -> Tuple[str, str, str]:
    eval_root = os.path.join(output_dir, "rfid_eval", f"step_{step}")
    return eval_root, os.path.join(eval_root, "gt"), os.path.join(eval_root, "recon")


def build_model(cfg, device: torch.device) -> DCSAE:
    return DCSAE(
        encoder_type=cfg.encoder.type,
        dinov3_model_dir=cfg.encoder.dinov3_model_dir,
        siglip2_model_name=cfg.encoder.siglip2_model_name,
        dinov2_model_name=cfg.encoder.dinov2_model_name,
        qwen3_vit_model_name=getattr(cfg.encoder, "qwen3_vit_model_name", "Qwen/Qwen3-VL-8B-Instruct"),
        internvl3_model_name=getattr(cfg.encoder, "internvl3_model_name", "OpenGVLab/InternVL3-8B"),
        internvl3_resize_target=getattr(cfg.encoder, "internvl3_resize_target", 224),
        image_size=cfg.data.image_size,
        in_channels=3,
        out_channels=3,
        hidden_size=cfg.model.hidden_size,
        hidden_size_x=cfg.model.hidden_size_x,
        # Decoder type selection
        decoder_type=getattr(cfg.model, "decoder_type", "flow_matching"),
        # Flow matching decoder config
        num_decoder_blocks=cfg.model.num_decoder_blocks,
        nerf_max_freqs=cfg.model.nerf_max_freqs,
        flow_steps=cfg.model.flow_steps,
        time_dim=cfg.model.time_dim,
        # ViT decoder config
        vit_decoder_hidden_size=getattr(cfg.model, "vit_decoder_hidden_size", 1024),
        vit_decoder_num_layers=getattr(cfg.model, "vit_decoder_num_layers", 24),
        vit_decoder_num_heads=getattr(cfg.model, "vit_decoder_num_heads", 16),
        vit_decoder_intermediate_size=getattr(cfg.model, "vit_decoder_intermediate_size", 4096),
        vit_decoder_dropout=getattr(cfg.model, "vit_decoder_dropout", 0.0),
        gradient_checkpointing=getattr(cfg.model, "gradient_checkpointing", False),
        # HF branch
        enable_hf_branch=cfg.model.enable_hf_branch,
        hf_dim=cfg.model.hf_dim,
        hf_encoder_type=getattr(cfg.model, "hf_encoder_type", "cnn"),
        hf_encoder_config_path=getattr(cfg.model, "hf_encoder_config_path", None),
        hf_encoder_patch_size=getattr(cfg.model, "hf_encoder_patch_size", None),
        hf_token_norm=getattr(cfg.model, "hf_token_norm", False),
        hf_dropout_prob=cfg.model.hf_dropout_prob,
        hf_noise_std=cfg.model.hf_noise_std,
        hf_noise_alpha_schedule=getattr(cfg.model, "hf_noise_alpha_schedule", "alpha_one"),
        hf_loss_weight=cfg.model.hf_loss_weight,
        recon_l2_weight=cfg.loss.recon_l2_weight,
        recon_l1_weight=cfg.loss.recon_l1_weight,
        recon_lpips_weight=cfg.loss.recon_lpips_weight,
        recon_gan_weight=cfg.loss.recon_gan_weight,
        lora_rank=cfg.model.lora_rank,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        enable_lora=cfg.model.enable_lora,
        target_latent_channels=cfg.model.target_latent_channels,
        variational=cfg.model.variational,
        kl_weight=cfg.model.kl_weight,
        skip_to_moments=cfg.model.skip_to_moments,
        noise_tau=cfg.model.noise_tau,
        random_masking_channel_ratio=cfg.model.random_masking_channel_ratio,
        denormalize_decoder_output=cfg.model.denormalize_decoder_output,
        # DeMerger (model-level)
        enable_de_merger=getattr(cfg.model, "enable_de_merger", False),
        de_merger_expand_size=getattr(cfg.model, "de_merger_expand_size", 2),
        de_merger_num_layers=getattr(cfg.model, "de_merger_num_layers", 2),
        de_merger_hidden_dim=getattr(cfg.model, "de_merger_hidden_dim", None),
        de_merger_nhead=getattr(cfg.model, "de_merger_nhead", 8),
        # Ablation
        zero_out_semantic=getattr(cfg.model, "zero_out_semantic", False),
    ).to(device)


def freeze_for_hf_and_pixel_decoder(
    model: DCSAE,
    freeze_semantic_encoder: bool = True,
    freeze_hf_encoder: bool = False,
):
    requires_grad(model, False)
    trainable_modules = [
        model.fused_norm,
        model.fused_proj,
        model.decoder,
    ]
    if getattr(model, "de_merger", None) is not None:
        trainable_modules.append(model.de_merger)
    if (not freeze_hf_encoder) and (getattr(model, "hf_encoder", None) is not None):
        trainable_modules.append(model.hf_encoder)
    encoder_trainable_merger = getattr(getattr(model, "encoder", None), "trainable_merger", None)
    if encoder_trainable_merger is not None:
        trainable_modules.append(encoder_trainable_merger)
    # Flow matching specific modules (may be None for ViT decoder)
    if model.t_embedder is not None:
        trainable_modules.append(model.t_embedder)
    if model.coord_embedder is not None:
        trainable_modules.append(model.coord_embedder)

    for module in trainable_modules:
        requires_grad(module, True)

    if not freeze_semantic_encoder:
        requires_grad(model.encoder, True)


def build_discriminator(cfg, device: torch.device):
    if not cfg.discriminator.enabled:
        return None, None, None

    disc_type = cfg.discriminator.disc_type.lower()
    if disc_type == "dino":
        key_depths = tuple(int(x) for x in cfg.discriminator.key_depths.split(","))
        discriminator = DinoDisc(
            device=device,
            dino_ckpt_path=cfg.discriminator.dino_ckpt_path,
            ks=cfg.discriminator.ks,
            key_depths=key_depths,
            norm_type=cfg.discriminator.norm,
            using_spec_norm=True,
            norm_eps=1e-6,
            recipe=cfg.discriminator.recipe,
        ).to(device)
        disc_loss_fn = hinge_d_loss
    elif disc_type == "patchgan":
        discriminator = NLayerDiscriminator(
            in_channels=3,
            ndf=cfg.discriminator.ndf,
            n_layers=cfg.discriminator.n_layers,
            norm=cfg.discriminator.norm,
        ).to(device)
        disc_loss_fn = d_hinge_loss
    else:
        raise ValueError(f"Unsupported discriminator type: {cfg.discriminator.disc_type}")

    diffaug = DiffAug(
        prob=cfg.discriminator.diffaug_prob,
        cutout=cfg.discriminator.diffaug_cutout,
    )
    return discriminator, diffaug, disc_loss_fn


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int,
    step: int,
    output_dir: str,
    rank: int,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
    compute_rfid: bool,
    fid_batch_size: int,
    fid_num_workers: int,
    keep_rfid_images: bool,
    logger=None,
) -> Tuple[float, float, Optional[float], int]:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_count = 0
    rfid_value: Optional[float] = None

    requested_max_batches = len(val_loader) if max_batches is None or max_batches <= 0 else max_batches
    local_num_batches_tensor = torch.tensor(len(val_loader), device=device, dtype=torch.long)
    dist.all_reduce(local_num_batches_tensor, op=dist.ReduceOp.MIN)
    actual_max_batches = min(requested_max_batches, local_num_batches_tensor.item())

    eval_root: Optional[str] = None
    gt_dir: Optional[str] = None
    recon_dir: Optional[str] = None
    if compute_rfid:
        eval_root, gt_dir, recon_dir = get_rfid_dirs(output_dir, step)
        assert gt_dir is not None and recon_dir is not None
        if rank == 0:
            os.makedirs(gt_dir, exist_ok=True)
            os.makedirs(recon_dir, exist_ok=True)
        dist.barrier()

    for idx, (x, _) in enumerate(val_loader):
        if idx >= actual_max_batches:
            break
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            out = model(x)
        recon = out.sample.float().clamp(-1.0, 1.0)
        x_f = x.float()
        per_sample_mse = F.mse_loss(recon, x_f, reduction="none").mean(dim=[1, 2, 3])
        batch_psnr_sum, batch_count = calculate_batch_psnr(recon, x_f)

        total_loss += per_sample_mse.sum().item()
        total_psnr += batch_psnr_sum
        total_count += batch_count

        if compute_rfid:
            assert gt_dir is not None and recon_dir is not None
            img_gt_save = torch.clamp((x_f + 1.0) / 2.0, 0, 1)
            img_recon_save = torch.clamp((recon + 1.0) / 2.0, 0, 1)
            for sample_idx in range(batch_count):
                file_name = f"r{rank}_b{idx}_{sample_idx}.png"
                save_image(img_gt_save[sample_idx], os.path.join(gt_dir, file_name))
                save_image(img_recon_save[sample_idx], os.path.join(recon_dir, file_name))

    metrics = torch.tensor([total_loss, total_psnr, float(total_count)], device=device, dtype=torch.float64)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    denom = max(metrics[2].item(), 1.0)
    global_count = int(metrics[2].item())

    if compute_rfid:
        dist.barrier()
        if rank == 0:
            if HAS_FID:
                try:
                    rfid_value = float(
                        fid_score.calculate_fid_given_paths(
                            paths=[gt_dir, recon_dir],
                            batch_size=fid_batch_size,
                            device=device,
                            dims=2048,
                            num_workers=fid_num_workers,
                        )
                    )
                except Exception as exc:
                    if logger is not None:
                        logger.error(f"rFID calculation failed at step {step}: {exc}")
                    rfid_value = None
            elif logger is not None:
                logger.warning("pytorch-fid not found. rFID evaluation will be skipped.")

        rfid_tensor = torch.tensor(
            [-1.0 if rfid_value is None else rfid_value],
            device=device,
            dtype=torch.float64,
        )
        dist.broadcast(rfid_tensor, src=0)
        rfid_value = None if rfid_tensor.item() < 0 else rfid_tensor.item()

        if rank == 0 and not keep_rfid_images:
            assert eval_root is not None
            shutil.rmtree(eval_root, ignore_errors=True)
        dist.barrier()

    model.train()
    return metrics[0].item() / denom, metrics[1].item() / denom, rfid_value, global_count


def try_load_checkpoint(model: DCSAE, ckpt_path: str, strict_load: bool, logger):
    if not ckpt_path:
        return 0, None
    if not os.path.isfile(ckpt_path):
        logger.warning(f"Checkpoint not found: {ckpt_path}")
        return 0, None

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in payload:
        state_dict = payload["model"]
    elif "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload

    if not strict_load:
        model_state = model.state_dict()
        filtered_state = {}
        skipped_shape_mismatch = []
        for k, v in state_dict.items():
            if k in model_state and hasattr(v, "shape") and hasattr(model_state[k], "shape"):
                if tuple(v.shape) != tuple(model_state[k].shape):
                    skipped_shape_mismatch.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                    continue
            filtered_state[k] = v
        if skipped_shape_mismatch:
            logger.warning(
                f"Skipping {len(skipped_shape_mismatch)} mismatched checkpoint keys "
                f"(showing up to 10): {skipped_shape_mismatch[:10]}"
            )
        state_dict = filtered_state

    missing, unexpected = model.load_state_dict(state_dict, strict=strict_load)
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)
    step = payload.get("step", 0) if isinstance(payload, dict) else 0
    if not strict_load:
        # For transfer initialization (e.g., resolution change), keep weights only and
        # start a fresh optimization trajectory from step 0.
        step = 0

    logger.info(f"Loaded checkpoint from {ckpt_path} (step={step})")
    if not strict_load:
        logger.info(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    return step, payload


def main():
    cfg = load_and_merge_config()

    output_dir = cfg.logging.output_dir
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    experiment_dir = os.path.join(output_dir, "experiment")

    rank, local_rank, _ = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    logger = create_logger(experiment_dir, rank, name="train_sae")

    set_seed(cfg.training.seed + rank)
    torch.backends.cudnn.benchmark = True

    if cfg.training.precision == "bf16" and (not torch.cuda.is_bf16_supported()):
        raise ValueError("precision=bf16 but current GPU does not support bfloat16.")
    use_bf16 = cfg.training.precision == "bf16"
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    train_dataset, val_dataset = build_datasets(
        image_size=cfg.data.image_size,
        train_path=cfg.data.train_path,
        val_path=cfg.data.val_path,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        sampler=train_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        sampler=val_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_model(cfg, device=device)
    sae_ckpt = cfg.checkpoint.sae_ckpt
    auto_resume = False
    if not sae_ckpt:
        # Auto-resume from the latest valid step_*.pth in output_dir
        import glob as _glob
        _search_dirs = [
            os.path.join(cfg.logging.output_dir, "checkpoints"),
            cfg.logging.output_dir,
        ]
        _ckpts = sorted(
            [p for d in _search_dirs for p in _glob.glob(os.path.join(d, "step_*.pth"))],
            key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
            reverse=True,
        )
        for _c in _ckpts:
            try:
                torch.load(_c, map_location="cpu", weights_only=False)
                sae_ckpt = _c
                auto_resume = True
                if rank == 0:
                    logger.info(f"Auto-resuming from latest valid checkpoint: {sae_ckpt}")
                break
            except Exception as _e:
                if rank == 0:
                    logger.warning(f"Skipping corrupted checkpoint {_c}: {_e}")
    # Auto-resume: non-strict model weights but restore step + optimizer state.
    # Explicit sae_ckpt in config: honour cfg.checkpoint.strict_load as-is.
    strict_load = False if auto_resume else cfg.checkpoint.strict_load
    resume_step, resume_payload = try_load_checkpoint(
        model, sae_ckpt, strict_load, logger
    )
    # For auto-resume, step is reset to 0 by try_load_checkpoint (non-strict path);
    # override it with the actual step stored in the payload.
    if auto_resume and resume_payload is not None:
        resume_step = resume_payload.get("step", 0)

    # Freeze semantic encoder and/or HF encoder according to config.
    freeze_semantic_encoder = getattr(cfg.training, "freeze_semantic_encoder", True)
    freeze_hf_encoder = getattr(cfg.training, "freeze_hf_encoder", False)
    freeze_for_hf_and_pixel_decoder(
        model,
        freeze_semantic_encoder=freeze_semantic_encoder,
        freeze_hf_encoder=freeze_hf_encoder,
    )
    model.train()

    if rank == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {trainable / 1e6:.3f}M / {total / 1e6:.3f}M")
        logger.info(f"Distributed mode: {cfg.training.distributed}")

    model = build_wrapped_model(model, cfg, local_rank)

    # TensorBoard setup
    tb_writer = None
    if rank == 0 and HAS_TENSORBOARD:
        tb_log_dir = os.path.join(experiment_dir, "tensorboard")
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard logging to: {tb_log_dir}")
    if rank == 0 and getattr(cfg.logging, "eval_rfid", True) and not HAS_FID:
        logger.warning("pytorch-fid not found. Training-time rFID evaluation will be skipped.")

    # Wandb setup
    wandb_run = None
    wandb_project = getattr(cfg.logging, "wandb_project", "")
    if rank == 0 and HAS_WANDB and wandb_project:
        wandb_name = getattr(cfg.logging, "wandb_name", "") or os.path.basename(output_dir)
        wandb_entity = getattr(cfg.logging, "wandb_entity", "") or None
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            config=_cfg_to_dict(cfg),
            dir=output_dir,
            resume="allow",
        )
        logger.info(f"Wandb logging to project={wandb_project}, run={wandb_name}")

    discriminator, diffaug, disc_loss_fn = build_discriminator(cfg, device=device)
    if discriminator is not None and resume_payload is not None and (strict_load or auto_resume):
        if "discriminator" in resume_payload:
            disc_state_dict = resume_payload["discriminator"]
            if not strict_load:
                discriminator_state = discriminator.state_dict()
                filtered_disc_state = {}
                skipped_disc_shape_mismatch = []
                for k, v in disc_state_dict.items():
                    if k in discriminator_state and hasattr(v, "shape") and hasattr(discriminator_state[k], "shape"):
                        if tuple(v.shape) != tuple(discriminator_state[k].shape):
                            skipped_disc_shape_mismatch.append(
                                (k, tuple(v.shape), tuple(discriminator_state[k].shape))
                            )
                            continue
                    filtered_disc_state[k] = v
                if skipped_disc_shape_mismatch:
                    logger.warning(
                        f"Skipping {len(skipped_disc_shape_mismatch)} mismatched discriminator keys "
                        f"(showing up to 10): {skipped_disc_shape_mismatch[:10]}"
                    )
                disc_state_dict = filtered_disc_state
            disc_missing, disc_unexpected = discriminator.load_state_dict(
                disc_state_dict, strict=strict_load
            )
            if not strict_load:
                logger.info(
                    f"Discriminator missing keys: {len(disc_missing)}, "
                    f"unexpected keys: {len(disc_unexpected)}"
                )
    if discriminator is not None:
        discriminator = build_wrapped_model(discriminator, cfg, local_rank)

    optimizer = build_optimizer_for_model(model, cfg)

    scheduler = None
    if cfg.optimizer.scheduler == "cosine":
        min_lr_ratio = cfg.optimizer.min_lr / cfg.optimizer.lr if cfg.optimizer.lr > 0 else 0.1
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            warmup_steps=cfg.optimizer.warmup_steps,
            total_steps=cfg.training.max_steps,
            min_lr_ratio=min_lr_ratio,
        )

    optimizer_disc = None
    scheduler_disc = None
    if discriminator is not None:
        optimizer_disc = torch.optim.AdamW(
            [p for p in discriminator.parameters() if p.requires_grad],
            lr=cfg.discriminator.lr,
            betas=(cfg.discriminator.betas0, cfg.discriminator.betas1),
            weight_decay=cfg.discriminator.weight_decay,
        )
        if cfg.optimizer.scheduler == "cosine":
            min_lr_ratio_disc = (
                cfg.optimizer.min_lr / cfg.discriminator.lr if cfg.discriminator.lr > 0 else 0.1
            )
            scheduler_disc = get_cosine_schedule_with_warmup(
                optimizer=optimizer_disc,
                warmup_steps=cfg.optimizer.warmup_steps,
                total_steps=cfg.training.max_steps,
                min_lr_ratio=min_lr_ratio_disc,
            )

    if resume_payload is not None:
        if strict_load or auto_resume:
            optimizer_restored = _load_optimizer_state_dict(
                model,
                optimizer,
                resume_payload.get("optimizer"),
                logger,
                "optimizer",
                rank,
            )
            if scheduler is not None and ("scheduler" in resume_payload) and (resume_payload["scheduler"] is not None):
                if optimizer_restored:
                    scheduler.load_state_dict(resume_payload["scheduler"])
                elif rank == 0:
                    logger.warning("Skipping scheduler restore because optimizer state was not restored.")
            optimizer_disc_restored = _load_optimizer_state_dict(
                discriminator,
                optimizer_disc,
                resume_payload.get("optimizer_disc"),
                logger,
                "discriminator optimizer",
                rank,
            )
            if (
                scheduler_disc is not None
                and ("scheduler_disc" in resume_payload)
                and (resume_payload["scheduler_disc"] is not None)
            ):
                if optimizer_disc_restored:
                    scheduler_disc.load_state_dict(resume_payload["scheduler_disc"])
                elif rank == 0:
                    logger.warning(
                        "Skipping discriminator scheduler restore because optimizer state was not restored."
                    )
        else:
            logger.info("strict_load=false: skip optimizer/scheduler/discriminator resume states.")

    step = resume_step
    epoch = 0

    # Two-stage training: if resuming past freeze_encoder_step, re-apply freeze
    freeze_encoder_step = getattr(cfg.training, "freeze_encoder_step", -1)
    if freeze_encoder_step > 0 and step >= freeze_encoder_step:
        if rank == 0:
            logger.info(
                f"[Resume] step={step} >= freeze_encoder_step={freeze_encoder_step}, "
                f"freezing encoder"
            )
        freeze_for_hf_and_pixel_decoder(
            unwrap_model(model),
            freeze_semantic_encoder=True,
            freeze_hf_encoder=False,
        )
        optimizer = build_optimizer_for_model(model, cfg)
        if scheduler is not None:
            scheduler = get_scheduler(cfg, optimizer, step)
        # Re-load optimizer/scheduler states from checkpoint if available
        if resume_payload is not None and (strict_load or auto_resume):
            optimizer_restored = _load_optimizer_state_dict(
                model,
                optimizer,
                resume_payload.get("optimizer"),
                logger,
                "optimizer after freeze",
                rank,
            )
            if (
                scheduler is not None
                and "scheduler" in resume_payload
                and resume_payload["scheduler"] is not None
            ):
                if optimizer_restored:
                    try:
                        scheduler.load_state_dict(resume_payload["scheduler"])
                    except Exception as exc:
                        if rank == 0:
                            logger.warning(f"Skipping scheduler restore after freeze: {exc}")
                elif rank == 0:
                    logger.warning(
                        "Skipping scheduler restore after freeze because optimizer state was not restored."
                    )

    # Create a fixed visualization batch (same across all evals)
    vis_batch = None
    if len(val_dataset) > 0:
        vis_generator = torch.Generator()
        vis_generator.manual_seed(42)
        vis_indices = torch.randperm(len(val_dataset), generator=vis_generator)[:min(8, len(val_dataset))]
        vis_images = [val_dataset[i][0] for i in vis_indices]
        vis_batch = torch.stack(vis_images, dim=0)
        if rank == 0:
            logger.info(f"Created fixed visualization batch with {vis_batch.shape[0]} images")

    if rank == 0:
        logger.info("Start training DCSAE with Flow Matching.")

    rfid_every = cfg.logging.rfid_every if getattr(cfg.logging, "rfid_every", 0) > 0 else cfg.logging.save_every
    rfid_every = max(1, rfid_every)

    while step < cfg.training.max_steps:
        train_sampler.set_epoch(epoch)

        for x, _ in train_loader:
            if step >= cfg.training.max_steps:
                break
            x = x.to(device, non_blocking=True)

            gan_active = (
                discriminator is not None
                and cfg.loss.recon_gan_weight > 0
                and step >= cfg.discriminator.start_step
            )

            # Two-stage training: freeze encoder at the configured step (fires once)
            if freeze_encoder_step > 0 and step == freeze_encoder_step:
                if rank == 0:
                    print(f"\n{'='*60}")
                    print(f"[Stage 2] step={step}: Freezing semantic encoder")
                    print(f"{'='*60}\n")
                freeze_for_hf_and_pixel_decoder(
                    unwrap_model(model),
                    freeze_semantic_encoder=True,
                    freeze_hf_encoder=False,
                )
                # Rebuild optimizer with only trainable params
                optimizer = build_optimizer_for_model(model, cfg)
                # Rebuild scheduler from current step
                if scheduler is not None:
                    scheduler = get_scheduler(cfg, optimizer, step)
                if rank == 0:
                    trainable_params = [p for p in model.parameters() if p.requires_grad]
                    n_trainable = sum(p.numel() for p in trainable_params)
                    n_total = sum(p.numel() for p in model.parameters())
                    print(f"  Trainable params: {n_trainable:,} / {n_total:,} "
                          f"({100*n_trainable/n_total:.1f}%)")

            if gan_active:
                discriminator.eval()
                requires_grad(discriminator, False)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_bf16):
                loss_dict = model(
                    x,
                    return_loss=True,
                    discriminator=discriminator if gan_active else None,
                    diffaug=diffaug if gan_active else None,
                    return_dict=True,
                )
                loss = loss_dict["loss"]
            loss.backward()

            # Compute gradient norm before optimizer step
            grad_norm = float(get_grad_norm(model))

            # Gradient clipping
            clip_grad_norm(model, max_norm=1.0)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            disc_loss = torch.tensor(0.0, device=device)
            d_real = torch.tensor(0.0, device=device)
            d_fake = torch.tensor(0.0, device=device)
            disc_grad_norm = 0.0
            if gan_active:
                discriminator.train()
                requires_grad(discriminator, True)
                optimizer_disc.zero_grad(set_to_none=True)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_bf16):
                    recon_detach = model(x).sample.detach().clamp(-1.0, 1.0)
                    # Keep discriminator input discretization same as VAE+GAN implementation.
                    recon_detach = torch.round((recon_detach + 1.0) * 127.5) / 127.5 - 1.0
                    x_aug = diffaug.aug(x) if diffaug is not None else x
                    recon_aug = diffaug.aug(recon_detach) if diffaug is not None else recon_detach

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_bf16):
                    logits_fake = discriminator(recon_aug)
                    logits_real = discriminator(x_aug)
                    disc_loss = disc_loss_fn(logits_real, logits_fake)
                    d_real = logits_real.mean().detach()
                    d_fake = logits_fake.mean().detach()
                disc_loss.backward()
                disc_grad_norm = float(get_grad_norm(discriminator))
                optimizer_disc.step()
                if scheduler_disc is not None:
                    scheduler_disc.step()

            step += 1

            # Reduce losses across ranks
            reduced_loss = reduce_mean_tensor(loss)
            reduced_flow_loss = reduce_mean_tensor(loss_dict["flow_loss"])
            reduced_recon_loss = reduce_mean_tensor(loss_dict["recon_loss"])
            reduced_l2_loss = reduce_mean_tensor(loss_dict["l2_loss"])
            reduced_l1_loss = reduce_mean_tensor(loss_dict["l1_loss"])
            reduced_lpips_loss = reduce_mean_tensor(loss_dict["lpips_loss"])
            reduced_gan_loss = reduce_mean_tensor(loss_dict["gan_loss"])
            reduced_disc_loss = reduce_mean_tensor(disc_loss)
            reduced_d_real = reduce_mean_tensor(d_real)
            reduced_d_fake = reduce_mean_tensor(d_fake)
            lr = optimizer.param_groups[0]["lr"]
            if gan_active:
                disc_lr = optimizer_disc.param_groups[0]["lr"]
            if step % cfg.logging.log_every == 0 and rank == 0:
                log_msg = (
                    f"Step {step} lr={lr:.2e}: total={reduced_loss.item():.4f} | "
                    f"flow={reduced_flow_loss.item():.4f} recon={reduced_recon_loss.item():.4f} "
                    f"(L2={reduced_l2_loss.item():.4f}, L1={reduced_l1_loss.item():.4f}, "
                    f"LPIPS={reduced_lpips_loss.item():.4f}) | grad_norm={grad_norm:.4f}"
                )
                if gan_active:
                    log_msg += (
                        f" | GAN_g={reduced_gan_loss.item():.4f} D_loss={reduced_disc_loss.item():.4f} "
                        f"D_real={reduced_d_real.item():.3f} D_fake={reduced_d_fake.item():.3f} "
                        f"disc_grad_norm={disc_grad_norm:.4f} disc_lr={disc_lr:.2e}"
                    )
                logger.info(log_msg)
                
                # TensorBoard logging
                if tb_writer is not None:
                    # Total loss
                    tb_writer.add_scalar("Loss/total", reduced_loss.item(), step)
                    tb_writer.add_scalar("Loss/flow", reduced_flow_loss.item(), step)
                    tb_writer.add_scalar("Loss/recon", reduced_recon_loss.item(), step)
                    
                    # Reconstruction loss components
                    tb_writer.add_scalar("Loss/L2", reduced_l2_loss.item(), step)
                    tb_writer.add_scalar("Loss/L1", reduced_l1_loss.item(), step)
                    tb_writer.add_scalar("Loss/LPIPS", reduced_lpips_loss.item(), step)
                    
                    # Gradient norms
                    tb_writer.add_scalar("GradNorm/generator", grad_norm, step)
                    
                    # Learning rate
                    tb_writer.add_scalar("LR/generator", lr, step)
                    
                    # GAN losses (when active)
                    if gan_active:
                        tb_writer.add_scalar("Loss/GAN_gen", reduced_gan_loss.item(), step)
                        tb_writer.add_scalar("Loss/GAN_disc", reduced_disc_loss.item(), step)
                        tb_writer.add_scalar("Discriminator/logits_real", reduced_d_real.item(), step)
                        tb_writer.add_scalar("Discriminator/logits_fake", reduced_d_fake.item(), step)
                        tb_writer.add_scalar("GradNorm/discriminator", disc_grad_norm, step)
                        tb_writer.add_scalar("LR/discriminator", disc_lr, step)

            should_run_rfid = getattr(cfg.logging, "eval_rfid", True) and step % rfid_every == 0

            if rank == 0 and wandb_run is not None:
                wandb_log = {
                    "loss/total": reduced_loss.item(),
                    "loss/flow": reduced_flow_loss.item(),
                    "loss/recon": reduced_recon_loss.item(),
                    "loss/L2": reduced_l2_loss.item(),
                    "loss/L1": reduced_l1_loss.item(),
                    "loss/LPIPS": reduced_lpips_loss.item(),
                    "grad_norm/generator": grad_norm,
                    "lr/generator": lr,
                }
                if gan_active:
                    wandb_log.update({
                        "loss/GAN_gen": reduced_gan_loss.item(),
                        "loss/GAN_disc": reduced_disc_loss.item(),
                        "discriminator/logits_real": reduced_d_real.item(),
                        "discriminator/logits_fake": reduced_d_fake.item(),
                        "grad_norm/discriminator": disc_grad_norm,
                        "lr/discriminator": disc_lr,
                    })
                # Two-stage indicator
                stage = 2 if (freeze_encoder_step > 0 and step >= freeze_encoder_step) else 1
                wandb_log["train/stage"] = stage
                wandb_run.log(wandb_log, step=step)

            should_run_rfid = getattr(cfg.logging, "eval_rfid", True) and step % rfid_every == 0

            if step % cfg.logging.eval_every == 0:
                val_loss, val_psnr, val_rfid, val_count = run_validation(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    max_batches=cfg.logging.val_max_batches,
                    step=step,
                    output_dir=output_dir,
                    rank=rank,
                    amp_dtype=amp_dtype,
                    amp_enabled=use_bf16,
                    compute_rfid=should_run_rfid,
                    fid_batch_size=cfg.logging.fid_batch_size,
                    fid_num_workers=cfg.logging.fid_num_workers,
                    keep_rfid_images=cfg.logging.keep_rfid_images,
                    logger=logger,
                )
                if rank == 0:
                    log_msg = (
                        f"[Eval {step}] images={val_count} "
                        f"val_loss={val_loss:.6f}, val_psnr={val_psnr:.4f}"
                    )
                    if should_run_rfid:
                        if val_rfid is not None:
                            log_msg += f", val_rfid={val_rfid:.4f}"
                        else:
                            log_msg += ", val_rfid=N/A"
                    logger.info(log_msg)
                    if tb_writer is not None:
                        tb_writer.add_scalar("Validation/loss", val_loss, step)
                        tb_writer.add_scalar("Validation/PSNR", val_psnr, step)
                        if should_run_rfid and val_rfid is not None:
                            tb_writer.add_scalar("Validation/rFID", val_rfid, step)

                    if wandb_run is not None:
                        wandb_eval_log = {"val/loss": val_loss, "val/PSNR": val_psnr}
                        if should_run_rfid and val_rfid is not None:
                            wandb_eval_log["val/rFID"] = val_rfid
                        wandb_run.log(wandb_eval_log, step=step)

                if vis_batch is not None:
                    vis_path = save_visualization(
                        model=model,
                        vis_batch=vis_batch,
                        step=step,
                        output_dir=output_dir,
                        device=device,
                        amp_dtype=amp_dtype,
                        amp_enabled=use_bf16,
                        rank=rank,
                        nrow=4,
                    )
                    if vis_path is not None:
                        logger.info(f"Saved visualization to {vis_path}")
                        if wandb_run is not None:
                            wandb_run.log(
                                {"vis/reconstruction": wandb.Image(vis_path)},
                                step=step,
                            )

            if step % cfg.logging.save_every == 0:
                model_state = get_model_state_dict(model)
                optimizer_state = get_optimizer_state_dict(model, optimizer)
                discriminator_state = (
                    get_model_state_dict(discriminator) if discriminator is not None else None
                )
                optimizer_disc_state = get_optimizer_state_dict(discriminator, optimizer_disc)
                if rank == 0:
                    os.makedirs(ckpt_dir, exist_ok=True)
                    ckpt_path = os.path.join(ckpt_dir, f"step_{step}.pth")
                    torch.save(
                        {
                            "model": model_state,
                            "step": step,
                            "optimizer": optimizer_state,
                            "scheduler": scheduler.state_dict() if scheduler is not None else None,
                            "discriminator": discriminator_state,
                            "optimizer_disc": optimizer_disc_state,
                            "scheduler_disc": scheduler_disc.state_dict() if scheduler_disc is not None else None,
                            "config": cfg,
                        },
                        ckpt_path,
                    )
                    logger.info(f"Saved checkpoint to {ckpt_path}")
                dist.barrier()

        epoch += 1

    # Close TensorBoard writer
    if tb_writer is not None:
        tb_writer.close()
    if wandb_run is not None:
        wandb_run.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()
