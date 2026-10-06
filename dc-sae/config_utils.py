import argparse
from dataclasses import dataclass, field
from typing import Optional

import yaml

from path_env_overrides import SAE_CONFIG_PATH_ENV_VARS, apply_path_env_overrides


@dataclass
class DataConfig:
    train_path: str = "datasets/imagenet/train"
    val_path: str = "datasets/imagenet/val"
    image_size: int = 256
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class EncoderConfig:
    type: str = "dinov2"
    dinov3_model_dir: str = "pretrained/dinov3"
    siglip2_model_name: str = "google/siglip2-base-patch16-256"
    dinov2_model_name: str = "facebook/dinov2-with-registers-base"
    qwen3_vit_model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    internvl3_model_name: str = "OpenGVLab/InternVL3-8B"
    internvl3_resize_target: int = 224


@dataclass
class ModelConfig:
    # Decoder type: "flow_matching" or "vit_decoder"
    decoder_type: str = "flow_matching"
    
    # Shared config
    hidden_size: int = 1152
    hidden_size_x: int = 64
    
    # Flow matching decoder config
    num_decoder_blocks: int = 12
    nerf_max_freqs: int = 8
    flow_steps: int = 25
    time_dim: int = 256
    
    # ViT decoder config (when decoder_type="vit_decoder")
    # Available presets: XL (1024, 24, 16, 4096), L (768, 16, 12, 3072), B (512, 8, 8, 2048)
    vit_decoder_hidden_size: int = 1024
    vit_decoder_num_layers: int = 24
    vit_decoder_num_heads: int = 16
    vit_decoder_intermediate_size: int = 4096
    vit_decoder_dropout: float = 0.0
    gradient_checkpointing: bool = False

    # HF branch config
    enable_hf_branch: bool = True
    hf_dim: int = 256
    hf_encoder_type: str = "cnn"
    hf_encoder_config_path: Optional[str] = None
    # Pixel patch the HF encoder pools to. None uses the decode patch size, which a
    # DeMerger halves; set it to keep a frozen encoder's original HF token grid.
    hf_encoder_patch_size: Optional[int] = None
    hf_token_norm: bool = False
    hf_dropout_prob: float = 0.4
    hf_noise_std: float = 0.1
    hf_noise_alpha_schedule: str = "alpha_one"
    hf_loss_weight: float = 0.1

    # Latent config
    target_latent_channels: Optional[int] = None
    variational: bool = False
    kl_weight: float = 1e-8
    skip_to_moments: bool = True
    
    # Regularization
    noise_tau: float = 0.0
    random_masking_channel_ratio: float = 0.0
    denormalize_decoder_output: bool = False

    # LoRA
    lora_rank: int = 0
    lora_alpha: int = 0
    lora_dropout: float = 0.0
    enable_lora: bool = False

    # Ablation
    zero_out_semantic: bool = False

    # DeMerger (model-level)
    enable_de_merger: bool = False
    de_merger_expand_size: int = 2
    de_merger_num_layers: int = 2
    de_merger_hidden_dim: Optional[int] = None
    de_merger_nhead: int = 8


@dataclass
class LossConfig:
    recon_l2_weight: float = 1.0
    recon_l1_weight: float = 0.0
    recon_lpips_weight: float = 0.0
    recon_gan_weight: float = 0.0


@dataclass
class DiscriminatorConfig:
    # General GAN config
    enabled: bool = False
    disc_type: str = "dino"  # "dino" | "patchgan"
    lr: float = 2e-4
    start_step: int = 0  # Step to start GAN training (0 = from beginning)
    weight_decay: float = 0.0
    betas0: float = 0.5
    betas1: float = 0.9

    # DinoDisc options (when disc_type="dino")
    dino_ckpt_path: str = "./dino_vit_small_patch8_224.pth"
    recipe: str = "S_8"  # "S_8", "S_16", "B_16"
    ks: int = 3  # Kernel size for head
    norm: str = "bn"  # "bn" | "gn"
    key_depths: str = "2,5,8,11"  # Feature extraction depths
    diffaug_prob: float = 1.0  # DiffAug probability
    diffaug_cutout: float = 0.2  # DiffAug cutout ratio

    # PatchGAN options (when disc_type="patchgan")
    ndf: int = 64  # Number of discriminator filters
    n_layers: int = 4  # Number of layers


@dataclass
class OptimizerConfig:
    lr: float = 2e-4
    min_lr: float = 2e-5
    betas0: float = 0.9
    betas1: float = 0.999
    weight_decay: float = 0.0
    warmup_steps: int = 0
    scheduler: str = "cosine"


@dataclass
class TrainingConfig:
    max_steps: int = 100000
    precision: str = "bf16"
    seed: int = 42
    distributed: str = "ddp"
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_use_orig_params: bool = True
    freeze_encoder_step: int = -1  # -1 = disabled; positive = auto-freeze encoder at this step


@dataclass
class LoggingConfig:
    output_dir: str = "results_sae/default"
    log_every: int = 50
    eval_every: int = 2000
    save_every: int = 5000
    val_max_batches: int = 200
    eval_rfid: bool = True
    rfid_every: int = 0  # <=0 means follow save_every
    fid_batch_size: int = 50
    fid_num_workers: int = 8
    keep_rfid_images: bool = False
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_name: str = ""


@dataclass
class CheckpointConfig:
    sae_ckpt: str = ""
    strict_load: bool = False


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


def _merge_dataclass(dc_obj, updates: dict):
    if updates is None:
        return dc_obj
    for key, value in updates.items():
        if not hasattr(dc_obj, key):
            continue
        field_value = getattr(dc_obj, key)
        if hasattr(field_value, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(field_value, value)
        else:
            setattr(dc_obj, key, value)
    return dc_obj


def load_config(config_path: str, *, use_env_paths: bool = False) -> TrainConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    apply_path_env_overrides(raw, SAE_CONFIG_PATH_ENV_VARS, enabled=use_env_paths)
    cfg = TrainConfig()
    _merge_dataclass(cfg, raw)
    return cfg


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DCSAE with Flow Matching")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")

    # Common runtime overrides.
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--val-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--scheduler", type=str, default=None, choices=["constant", "cosine"])
    parser.add_argument("--precision", type=str, default=None, choices=["fp32", "bf16"])
    parser.add_argument("--distributed", type=str, default=None, choices=["ddp", "fsdp"])
    parser.add_argument("--fsdp-sharding-strategy", type=str, default=None, choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"])
    parser.add_argument("--fsdp-use-orig-params", type=str, default=None, choices=["true", "false"])

    # GAN / Discriminator overrides
    parser.add_argument("--gan-start-step", type=int, default=None, 
                        help="Step to start GAN training")
    parser.add_argument("--disc-lr", type=float, default=None,
                        help="Discriminator learning rate")

    parser.add_argument("--sae-ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--use-env-paths",
        action="store_true",
        help="Prefer known path environment variables over config and CLI path values.",
    )
    return parser


def load_and_merge_config(cli_args=None) -> TrainConfig:
    parser = get_args_parser()
    args = parser.parse_args(cli_args)

    cfg = load_config(args.config, use_env_paths=args.use_env_paths)

    if args.train_path is not None:
        cfg.data.train_path = args.train_path
    if args.val_path is not None:
        cfg.data.val_path = args.val_path
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.image_size is not None:
        cfg.data.image_size = args.image_size
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

    if args.output_dir is not None:
        cfg.logging.output_dir = args.output_dir
    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps
    if args.lr is not None:
        cfg.optimizer.lr = args.lr
    if args.min_lr is not None:
        cfg.optimizer.min_lr = args.min_lr
    if args.warmup_steps is not None:
        cfg.optimizer.warmup_steps = args.warmup_steps
    if args.scheduler is not None:
        cfg.optimizer.scheduler = args.scheduler
    if args.precision is not None:
        cfg.training.precision = args.precision
    if args.distributed is not None:
        cfg.training.distributed = args.distributed
    if args.fsdp_sharding_strategy is not None:
        cfg.training.fsdp_sharding_strategy = args.fsdp_sharding_strategy
    if args.fsdp_use_orig_params is not None:
        cfg.training.fsdp_use_orig_params = args.fsdp_use_orig_params == "true"

    # GAN / Discriminator overrides
    if args.gan_start_step is not None:
        cfg.discriminator.start_step = args.gan_start_step
    if args.disc_lr is not None:
        cfg.discriminator.lr = args.disc_lr
    
    if args.sae_ckpt is not None:
        cfg.checkpoint.sae_ckpt = args.sae_ckpt
    if args.seed is not None:
        cfg.training.seed = args.seed

    final_env_overrides = apply_path_env_overrides(
        cfg,
        SAE_CONFIG_PATH_ENV_VARS,
        enabled=args.use_env_paths,
    )
    if final_env_overrides:
        print("Path environment overrides enabled for SAE config:")
        for item in final_env_overrides:
            print(f"  {item['field']} <= ${item['env']} -> {item['value']}")

    return cfg
