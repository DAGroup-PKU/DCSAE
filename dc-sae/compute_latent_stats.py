"""
Compute mean and std of dc-sae latent features on ImageNet.
Useful for DiT training normalization.

Usage:
torchrun --nproc_per_node=8 dc-sae/compute_latent_stats.py \
    --config dc-sae/dinov2_base_sae_vit_decoder.yaml \
    --ckpt results_sae/xxx/step_xxx.pth \
    --data-path /path/to/imagenet/train \
    --output-dir results_sae/latent_stats
"""

import argparse
import math
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config_utils import load_config
from model import DCSAE
from models.rae.utils.ddp_utils import cleanup_ddp, setup_ddp
from models.rae.utils.image_utils import center_crop_arr


# ==========================================
#         Running Statistics
# ==========================================

class RunningStats:
    """
    Welford's online algorithm for computing running mean and variance.
    Numerically stable for large datasets.
    """
    def __init__(self, shape, device='cpu'):
        self.n = 0
        self.mean = torch.zeros(shape, device=device, dtype=torch.float64)
        self.M2 = torch.zeros(shape, device=device, dtype=torch.float64)
    
    def update_batch(self, x: torch.Tensor):
        """
        Batch update for efficiency.
        x: [B, C, H, W]
        """
        if x.dim() == 4:
            B, C, H, W = x.shape
            batch_mean = x.mean(dim=[0, 2, 3]).double()  # [C]
            batch_var = x.var(dim=[0, 2, 3], unbiased=False).double()  # [C]
            batch_n = B * H * W
        elif x.dim() == 3:
            # x: [B, N, C] -> [B, C, H, W] where N = H * W
            B, N, C = x.shape
            batch_mean = x.mean(dim=[0, 1]).double()  # [C]
            batch_var = x.var(dim=[0, 1], unbiased=False).double()  # [C]
            batch_n = B * N
        else:
            raise ValueError(f"Unexpected tensor shape: {x.shape}")
        
        if self.n == 0:
            self.mean = batch_mean
            self.M2 = batch_var * batch_n
            self.n = batch_n
        else:
            delta = batch_mean - self.mean
            total_n = self.n + batch_n
            self.mean = self.mean + delta * batch_n / total_n
            self.M2 = self.M2 + batch_var * batch_n + delta ** 2 * self.n * batch_n / total_n
            self.n = total_n
    
    def get_mean(self):
        return self.mean.float()
    
    def get_std(self):
        if self.n < 2:
            return torch.zeros_like(self.mean).float()
        variance = self.M2 / self.n
        return torch.sqrt(variance).float()
    
    def get_var(self):
        if self.n < 2:
            return torch.zeros_like(self.mean).float()
        return (self.M2 / self.n).float()


# ==========================================
#              Model Building
# ==========================================

from train_sae import build_model as build_dc_sae


from checkpoint_utils import load_sae_checkpoint


@torch.no_grad()
def encode_with_dc_sae(sae_model: DCSAE, x_img: torch.Tensor):
    """
    Encode images using dc-sae.
    Returns:
        fused_pre: [B, C, H, W] fused latent before fused_norm
        fused_post: [B, C, H, W] fused latent after fused_norm
        z: [B, semantic_channels, H_p, W_p] semantic latent before fusion
    """
    sae_model.eval()

    enc = sae_model._infer_latent(x_img)
    z = enc.latent  # [B, semantic_channels, H_p, W_p]

    s_sem = sae_model._encoder_semantic(z)  # [B, N, semantic_channels]
    B, N, _ = s_sem.shape
    s_h, s_w = z.shape[-2:]
    s_hf = sae_model._extract_hf_tokens(
        x_img=x_img,
        bsz=B,
        s_h=s_h,
        s_w=s_w,
        num_patches=N,
        device=z.device,
        dtype=s_sem.dtype,
        force_drop_hf=False,
    )

    fused_pre_tokens = torch.cat([s_sem, s_hf], dim=-1)
    fused_post_tokens = sae_model.fused_norm(fused_pre_tokens)

    H = W = int(math.sqrt(N))
    fused_pre = fused_pre_tokens.transpose(1, 2).contiguous().view(B, -1, H, W)
    fused_post = fused_post_tokens.transpose(1, 2).contiguous().view(B, -1, H, W)

    return fused_pre, fused_post, z


# ==========================================
#                   Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Compute latent statistics for dc-sae")
    
    # Data
    parser.add_argument("--data-path", type=str, required=True, help="Path to ImageNet train set.")
    parser.add_argument("--num-samples", type=int, default=50000, help="Number of samples to use.")
    
    # Model
    parser.add_argument("--config", type=str, required=True, help="dc-sae config path")
    parser.add_argument("--ckpt", type=str, required=True, help="dc-sae checkpoint path")
    
    # Processing
    parser.add_argument("--batch-size", type=int, default=64)
    
    # Output
    parser.add_argument("--output-dir", type=str, default="results_sae/latent_stats")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    # 1. Initialize DDP
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # Set seed
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    # Load config
    sae_cfg = load_config(args.config)
    image_size = sae_cfg.data.image_size

    # Create output directory
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print("=" * 60)
        print(" Computing Latent Statistics for dc-sae")
        print("=" * 60)
        print(f" Config:      {args.config}")
        print(f" Checkpoint:  {args.ckpt}")
        print(f" Image Size:  {image_size}")
        print(f" Num Samples: {args.num_samples}")
        print(f" GPUs:        {world_size}")
        print("=" * 60)
    
    dist.barrier()

    # 2. Load dc-sae
    if rank == 0:
        print("\nLoading dc-sae...")
    
    sae_model = build_dc_sae(sae_cfg, device)
    load_sae_checkpoint(sae_model, args.ckpt, device, verbose=(rank == 0))
    sae_model.eval()
    
    semantic_channels = sae_model.semantic_channels
    hf_dim = sae_model.hf_dim
    total_channels = semantic_channels + hf_dim
    
    if rank == 0:
        print(f"Semantic channels: {semantic_channels}, HF dim: {hf_dim}, Total: {total_channels}")
    
    # 3. Load dataset
    if rank == 0:
        print("\nLoading dataset...")
    
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),
    ])
    
    dataset = ImageFolder(args.data_path, transform=transform)
    
    if rank == 0:
        print(f"Total dataset size: {len(dataset)}")
    
    # Random subset
    if args.num_samples < len(dataset):
        torch.manual_seed(args.seed)
        indices = torch.randperm(len(dataset))[:args.num_samples].tolist()
        dataset = Subset(dataset, indices)
    
    if rank == 0:
        print(f"Using {len(dataset)} samples")
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    # 4. Compute statistics
    if rank == 0:
        print("\nComputing statistics...")
    
    # Initialize running stats for raw semantic latent z
    stats_z = RunningStats(shape=(semantic_channels,), device=device)

    # Initialize running stats for fused latent before fused_norm
    stats_pre_full = RunningStats(shape=(total_channels,), device=device)
    stats_pre_semantic = RunningStats(shape=(semantic_channels,), device=device)
    stats_pre_hf = RunningStats(shape=(hf_dim,), device=device)

    # Legacy / backward-compatible stats: post-fused_norm fused latent
    stats_full = RunningStats(shape=(total_channels,), device=device)
    stats_semantic = RunningStats(shape=(semantic_channels,), device=device)
    stats_hf = RunningStats(shape=(hf_dim,), device=device)
    
    # Track min/max
    global_min = torch.full((total_channels,), float('inf'), device=device)
    global_max = torch.full((total_channels,), float('-inf'), device=device)
    
    with torch.no_grad():
        if rank == 0:
            iterator = tqdm(loader, desc="Processing")
        else:
            iterator = loader
        
        for x, _ in iterator:
            x = x.to(device)
            
            # Encode
            fused_pre, fused_post, z = encode_with_dc_sae(sae_model, x)

            # Raw semantic latent z stats
            stats_z.update_batch(z)

            # Pre-norm fused stats
            stats_pre_full.update_batch(fused_pre)
            pre_semantic_part = fused_pre[:, :semantic_channels, :, :]
            pre_hf_part = fused_pre[:, semantic_channels:, :, :]
            stats_pre_semantic.update_batch(pre_semantic_part)
            stats_pre_hf.update_batch(pre_hf_part)

            # Legacy / post-norm fused stats
            stats_full.update_batch(fused_post)
            semantic_part = fused_post[:, :semantic_channels, :, :]
            hf_part = fused_post[:, semantic_channels:, :, :]
            stats_semantic.update_batch(semantic_part)
            stats_hf.update_batch(hf_part)

            # Update min/max on post-norm fused latent for backward compatibility
            batch_min = fused_post.amin(dim=[0, 2, 3])
            batch_max = fused_post.amax(dim=[0, 2, 3])
            global_min = torch.minimum(global_min, batch_min)
            global_max = torch.maximum(global_max, batch_max)
    
    # 5. Synchronize across GPUs
    if rank == 0:
        print("\nSynchronizing across GPUs...")
    
    def sync_stats(stats):
        """Synchronize RunningStats across all ranks."""
        local_n = torch.tensor([stats.n], device=device, dtype=torch.float64)
        local_mean = stats.mean.clone()
        local_M2 = stats.M2.clone()
        
        all_n = [torch.zeros_like(local_n) for _ in range(world_size)]
        all_mean = [torch.zeros_like(local_mean) for _ in range(world_size)]
        all_M2 = [torch.zeros_like(local_M2) for _ in range(world_size)]
        
        dist.all_gather(all_n, local_n)
        dist.all_gather(all_mean, local_mean)
        dist.all_gather(all_M2, local_M2)
        
        combined_n = all_n[0].item()
        combined_mean = all_mean[0].clone()
        combined_M2 = all_M2[0].clone()
        
        for i in range(1, world_size):
            n_a = combined_n
            n_b = all_n[i].item()
            if n_b == 0:
                continue
            
            mean_a = combined_mean
            mean_b = all_mean[i]
            M2_a = combined_M2
            M2_b = all_M2[i]
            
            combined_n = n_a + n_b
            delta = mean_b - mean_a
            combined_mean = mean_a + delta * n_b / combined_n
            combined_M2 = M2_a + M2_b + delta ** 2 * n_a * n_b / combined_n
        
        final_mean = combined_mean.float()
        final_var = (combined_M2 / combined_n).float()
        final_std = torch.sqrt(final_var)
        
        return final_mean, final_std, final_var, combined_n
    
    # Sync all stats
    z_mean, z_std, _, _ = sync_stats(stats_z)
    pre_full_mean, pre_full_std, pre_full_var, _ = sync_stats(stats_pre_full)
    pre_semantic_mean, pre_semantic_std, _, _ = sync_stats(stats_pre_semantic)
    pre_hf_mean, pre_hf_std, _, _ = sync_stats(stats_pre_hf)
    full_mean, full_std, full_var, total_n = sync_stats(stats_full)
    semantic_mean, semantic_std, _, _ = sync_stats(stats_semantic)
    hf_mean, hf_std, _, _ = sync_stats(stats_hf)
    has_hf = hf_dim > 0
    
    # Sync min/max
    dist.all_reduce(global_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
    
    # 6. Save results (rank 0 only)
    if rank == 0:
        spatial_size = int(math.sqrt(total_n / args.num_samples))
        
        print("\n" + "=" * 60)
        print(" Statistics Summary")
        print("=" * 60)
        print(f" Total samples processed: {args.num_samples}")
        print(f" Total tokens processed:  {int(total_n)}")
        print(f" Total channels:          {total_channels}")
        print(f"   - Semantic:            {semantic_channels}")
        print(f"   - HF:                  {hf_dim}")
        print(f" Spatial size:            {spatial_size}x{spatial_size}")
        print("-" * 60)
        print(" Raw Semantic Latent z (before _encoder_semantic / fusion):")
        print(f"   Mean range: [{z_mean.min().item():.6f}, {z_mean.max().item():.6f}]")
        print(f"   Std range:  [{z_std.min().item():.6f}, {z_std.max().item():.6f}]")
        print(f"   Mean of mean: {z_mean.mean().item():.6f}")
        print(f"   Mean of std:  {z_std.mean().item():.6f}")
        print("-" * 60)
        print(" Pre-Norm Fused Latent (concat of semantic + HF):")
        print(f"   Mean range: [{pre_full_mean.min().item():.6f}, {pre_full_mean.max().item():.6f}]")
        print(f"   Std range:  [{pre_full_std.min().item():.6f}, {pre_full_std.max().item():.6f}]")
        print(f"   Mean of mean: {pre_full_mean.mean().item():.6f}")
        print(f"   Mean of std:  {pre_full_std.mean().item():.6f}")
        print("-" * 60)
        print(" Pre-Norm Semantic Part:")
        print(f"   Mean range: [{pre_semantic_mean.min().item():.6f}, {pre_semantic_mean.max().item():.6f}]")
        print(f"   Std range:  [{pre_semantic_std.min().item():.6f}, {pre_semantic_std.max().item():.6f}]")
        print(f"   Mean of mean: {pre_semantic_mean.mean().item():.6f}")
        print(f"   Mean of std:  {pre_semantic_std.mean().item():.6f}")
        print("-" * 60)
        print(" Pre-Norm HF Part:")
        if has_hf:
            print(f"   Mean range: [{pre_hf_mean.min().item():.6f}, {pre_hf_mean.max().item():.6f}]")
            print(f"   Std range:  [{pre_hf_std.min().item():.6f}, {pre_hf_std.max().item():.6f}]")
            print(f"   Mean of mean: {pre_hf_mean.mean().item():.6f}")
            print(f"   Mean of std:  {pre_hf_std.mean().item():.6f}")
        else:
            print("   Skipped (hf_dim=0)")
        print("-" * 60)
        print(" Post-Norm Fused Latent (legacy output):")
        print(f"   Mean range: [{full_mean.min().item():.6f}, {full_mean.max().item():.6f}]")
        print(f"   Std range:  [{full_std.min().item():.6f}, {full_std.max().item():.6f}]")
        print(f"   Mean of mean: {full_mean.mean().item():.6f}")
        print(f"   Mean of std:  {full_std.mean().item():.6f}")
        print("-" * 60)
        print(" Post-Norm Semantic Part (legacy semantic slice):")
        print(f"   Mean range: [{semantic_mean.min().item():.6f}, {semantic_mean.max().item():.6f}]")
        print(f"   Std range:  [{semantic_std.min().item():.6f}, {semantic_std.max().item():.6f}]")
        print(f"   Mean of mean: {semantic_mean.mean().item():.6f}")
        print(f"   Mean of std:  {semantic_std.mean().item():.6f}")
        print("-" * 60)
        print(" Post-Norm HF Part (legacy HF slice):")
        if has_hf:
            print(f"   Mean range: [{hf_mean.min().item():.6f}, {hf_mean.max().item():.6f}]")
            print(f"   Std range:  [{hf_std.min().item():.6f}, {hf_std.max().item():.6f}]")
            print(f"   Mean of mean: {hf_mean.mean().item():.6f}")
            print(f"   Mean of std:  {hf_std.mean().item():.6f}")
        else:
            print("   Skipped (hf_dim=0)")
        print("-" * 60)
        print(f" Global min: {global_min.min().item():.6f}")
        print(f" Global max: {global_max.max().item():.6f}")
        print("=" * 60)
        
        # Save as numpy
        np.savez(
            os.path.join(args.output_dir, "latent_stats.npz"),
            # Legacy / post-norm full latent
            mean=full_mean.cpu().numpy(),
            std=full_std.cpu().numpy(),
            var=full_var.cpu().numpy(),
            # Legacy / post-norm semantic + HF slices
            semantic_mean=semantic_mean.cpu().numpy(),
            semantic_std=semantic_std.cpu().numpy(),
            hf_mean=hf_mean.cpu().numpy(),
            hf_std=hf_std.cpu().numpy(),
            # Raw semantic latent before fusion
            z_mean=z_mean.cpu().numpy(),
            z_std=z_std.cpu().numpy(),
            # Pre-norm fused latent
            pre_mean=pre_full_mean.cpu().numpy(),
            pre_std=pre_full_std.cpu().numpy(),
            pre_var=pre_full_var.cpu().numpy(),
            pre_semantic_mean=pre_semantic_mean.cpu().numpy(),
            pre_semantic_std=pre_semantic_std.cpu().numpy(),
            pre_hf_mean=pre_hf_mean.cpu().numpy(),
            pre_hf_std=pre_hf_std.cpu().numpy(),
            # Min/Max of legacy post-norm full latent
            min=global_min.cpu().numpy(),
            max=global_max.cpu().numpy(),
            # Meta
            num_samples=args.num_samples,
            num_tokens=int(total_n),
            semantic_channels=semantic_channels,
            hf_dim=hf_dim,
        )
        print(f"\nSaved: {os.path.join(args.output_dir, 'latent_stats.npz')}")
        
        # Save as YAML
        stats_dict = {
            "config": args.config,
            "checkpoint": args.ckpt,
            "num_samples": args.num_samples,
            "image_size": image_size,
            "spatial_size": spatial_size,
            "total_channels": total_channels,
            "semantic_channels": semantic_channels,
            "hf_dim": hf_dim,
            "raw_semantic_latent": {
                "mean": z_mean.cpu().tolist(),
                "std": z_std.cpu().tolist(),
                "mean_of_mean": float(z_mean.mean().item()),
                "mean_of_std": float(z_std.mean().item()),
            },
            "pre_norm": {
                "full": {
                    "mean": pre_full_mean.cpu().tolist(),
                    "std": pre_full_std.cpu().tolist(),
                    "mean_of_mean": float(pre_full_mean.mean().item()),
                    "mean_of_std": float(pre_full_std.mean().item()),
                },
                "semantic": {
                    "mean": pre_semantic_mean.cpu().tolist(),
                    "std": pre_semantic_std.cpu().tolist(),
                    "mean_of_mean": float(pre_semantic_mean.mean().item()),
                    "mean_of_std": float(pre_semantic_std.mean().item()),
                },
                "hf": (
                    {
                        "mean": pre_hf_mean.cpu().tolist(),
                        "std": pre_hf_std.cpu().tolist(),
                        "mean_of_mean": float(pre_hf_mean.mean().item()),
                        "mean_of_std": float(pre_hf_std.mean().item()),
                    }
                    if has_hf
                    else {
                        "mean": [],
                        "std": [],
                        "mean_of_mean": 0.0,
                        "mean_of_std": 0.0,
                    }
                ),
            },
            "post_norm": {
                "full": {
                    "mean": full_mean.cpu().tolist(),
                    "std": full_std.cpu().tolist(),
                    "mean_of_mean": float(full_mean.mean().item()),
                    "mean_of_std": float(full_std.mean().item()),
                },
                "semantic": {
                    "mean": semantic_mean.cpu().tolist(),
                    "std": semantic_std.cpu().tolist(),
                    "mean_of_mean": float(semantic_mean.mean().item()),
                    "mean_of_std": float(semantic_std.mean().item()),
                },
                "hf": (
                    {
                        "mean": hf_mean.cpu().tolist(),
                        "std": hf_std.cpu().tolist(),
                        "mean_of_mean": float(hf_mean.mean().item()),
                        "mean_of_std": float(hf_std.mean().item()),
                    }
                    if has_hf
                    else {
                        "mean": [],
                        "std": [],
                        "mean_of_mean": 0.0,
                        "mean_of_std": 0.0,
                    }
                ),
            },
            # Legacy keys preserved for backward compatibility (post-norm semantics)
            "full": {
                "mean": full_mean.cpu().tolist(),
                "std": full_std.cpu().tolist(),
                "mean_of_mean": float(full_mean.mean().item()),
                "mean_of_std": float(full_std.mean().item()),
            },
            "semantic": {
                "mean": semantic_mean.cpu().tolist(),
                "std": semantic_std.cpu().tolist(),
                "mean_of_mean": float(semantic_mean.mean().item()),
                "mean_of_std": float(semantic_std.mean().item()),
            },
            "hf": (
                {
                    "mean": hf_mean.cpu().tolist(),
                    "std": hf_std.cpu().tolist(),
                    "mean_of_mean": float(hf_mean.mean().item()),
                    "mean_of_std": float(hf_std.mean().item()),
                }
                if has_hf
                else {
                    "mean": [],
                    "std": [],
                    "mean_of_mean": 0.0,
                    "mean_of_std": 0.0,
                }
            ),
            # Global range
            "global_min": float(global_min.min().item()),
            "global_max": float(global_max.max().item()),
        }
        
        yaml_path = os.path.join(args.output_dir, "latent_stats.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(stats_dict, f, default_flow_style=False, allow_unicode=True)
        print(f"Saved: {yaml_path}")
        
        # Save as PyTorch tensors
        torch.save({
            # Legacy / post-norm fields
            "mean": full_mean.cpu(),
            "std": full_std.cpu(),
            "var": full_var.cpu(),
            "semantic_mean": semantic_mean.cpu(),
            "semantic_std": semantic_std.cpu(),
            "hf_mean": hf_mean.cpu(),
            "hf_std": hf_std.cpu(),
            # Raw semantic latent before fusion
            "z_mean": z_mean.cpu(),
            "z_std": z_std.cpu(),
            # Pre-norm fused fields
            "pre_mean": pre_full_mean.cpu(),
            "pre_std": pre_full_std.cpu(),
            "pre_var": pre_full_var.cpu(),
            "pre_semantic_mean": pre_semantic_mean.cpu(),
            "pre_semantic_std": pre_semantic_std.cpu(),
            "pre_hf_mean": pre_hf_mean.cpu(),
            "pre_hf_std": pre_hf_std.cpu(),
            # Legacy min/max on post-norm fused latent
            "min": global_min.cpu(),
            "max": global_max.cpu(),
        }, os.path.join(args.output_dir, "latent_stats.pt"))
        print(f"Saved: {os.path.join(args.output_dir, 'latent_stats.pt')}")
        
        print("\n" + "=" * 60)
        print(" Normalization Suggestion for DiT Training")
        print("=" * 60)
        print(f" Raw semantic latent z normalization:")
        print(f"   z_raw_norm = (z - {z_mean.mean().item():.4f}) / {z_std.mean().item():.4f}")
        print("-" * 60)
        print(f" Pre-norm fused latent normalization:")
        print(f"   z_pre_norm = (z - {pre_full_mean.mean().item():.4f}) / {pre_full_std.mean().item():.4f}")
        print(f"   semantic_pre: (z[:, :{semantic_channels}] - {pre_semantic_mean.mean().item():.4f}) / {pre_semantic_std.mean().item():.4f}")
        if has_hf:
            print(f"   hf_pre:       (z[:, {semantic_channels}:] - {pre_hf_mean.mean().item():.4f}) / {pre_hf_std.mean().item():.4f}")
        else:
            print(f"   hf_pre:       skipped (hf_dim=0)")
        print("-" * 60)
        print(f" Post-norm fused latent normalization (legacy):")
        print(f"   z_norm = (z - {full_mean.mean().item():.4f}) / {full_std.mean().item():.4f}")
        print(f"   semantic: (z[:, :{semantic_channels}] - {semantic_mean.mean().item():.4f}) / {semantic_std.mean().item():.4f}")
        if has_hf:
            print(f"   hf:       (z[:, {semantic_channels}:] - {hf_mean.mean().item():.4f}) / {hf_std.mean().item():.4f}")
        else:
            print(f"   hf:       skipped (hf_dim=0)")
        print("=" * 60)
    
    dist.barrier()
    cleanup_ddp()
    
    if rank == 0:
        print("\nDone!")


if __name__ == "__main__":
    main()
