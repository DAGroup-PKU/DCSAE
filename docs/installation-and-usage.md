# Installation & Usage

[← Project overview](../README.md) · [简体中文](installation-and-usage-zn.md)

<a id="installation"></a>

## Installation

Use Python 3.10+ (3.12 used for the included CPU checks), Linux and CUDA GPUs for
training/distributed generation. Create and activate your own virtual environment.
Install a matching CUDA-enabled `torch`/`torchvision` pair for your machine, then:

```bash
python -m pip install -r requirements.txt
# Optional: W&B logging and IS / precision / recall
python -m pip install -r requirements-optional.txt
```

Run all commands **from the repository root**, using your active Python environment.
Training and distributed generation use standard `torchrun`; there is no dependency
on tmux, SSH automation, a specific Conda installation, or a cluster launcher.
Use a POSIX filesystem for code and outputs; object-storage mounts may not support
all file operations required by training.

`requirements.txt` declares dependency ranges; `requirements-tested.txt` records the
versions used for local import/CPU validation. That is not a CUDA benchmark or a
claim that all encoder families were tested end to end.

<a id="assets"></a>

## Data and pretrained models

Use ImageFolder layout, including class directories for the validation split:

```text
your_data/imagenet/train/<class_id>/*.JPEG
your_data/imagenet/val/<class_id>/*.JPEG
```

Set SAE data paths in its YAML or with `--train-path` / `--val-path`;
pass the DiT training dataset with `--data-path`. Train and validation class
ordering must agree. Default generation examples assume 1,000 ImageNet classes.
Datasets and weights are not stored in this Git repository. Download the released
SAE/DiT models from Hugging Face as described below.

<a id="download-models"></a>

### Download the released DC-SAE models

The official [DAGroup-PKU/DCSAE model repository](https://huggingface.co/DAGroup-PKU/DCSAE)
provides the following files for **each** resolution:

| Directory | Files | Latent layout |
| --- | --- | --- |
| `256/` | `sae.pt`, `dit.pt`, `latent_stats.pt`, `sae.yaml`, `dit.yaml` | 832 channels, 8×8 grid, 2× demerger |
| `512/` | `sae.pt`, `dit.pt`, `latent_stats.pt`, `sae.yaml`, `dit.yaml` | 1,024 channels, 16×16 grid, no demerger |

Run the following from the **repository root**. It downloads both model pairs
(about 10.4 GB total), keeps the original YAMLs beside the weights, and writes
local-path versions to `your_configs/`. Download DINOv2 separately using the next
section. These `your_*` directories are the literal destinations of the commands;
you can keep them or change the paths consistently.

```bash
python -m pip install -U huggingface_hub PyYAML
python - <<'PYTHON'
from pathlib import Path
from huggingface_hub import snapshot_download
import yaml

# Use ["256"] or ["512"] to download only one resolution.
resolutions = ["256", "512"]
snapshot_download(
    repo_id="DAGroup-PKU/DCSAE",
    local_dir="your_weights",
    allow_patterns=[f"{size}/*" for size in resolutions],
)
for size in resolutions:
    weights = Path("your_weights") / size
    configs = Path("your_configs") / size
    configs.mkdir(parents=True, exist_ok=True)
    for name in ("sae", "dit"):
        config = yaml.safe_load((weights / f"{name}.yaml").read_text())
        if name == "sae":
            config["encoder"]["dinov2_model_name"] = "your_pretrained/dinov2-with-registers-base"
            config["data"]["train_path"] = "your_data/imagenet/train"
            config["data"]["val_path"] = "your_data/imagenet/val"
            config["checkpoint"]["sae_ckpt"] = str(weights / "sae.pt")
            config["logging"]["output_dir"] = f"your_results/sae_{size}"
        else:
            config["misc"]["latent_stats_path"] = str(weights / "latent_stats.pt")
        (configs / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Ready: {weights} and {configs}")
PYTHON
```

The resulting files match the PSNR/rFID and gFID commands below without moving
weights manually. For example, the 512px gFID command reads
`your_weights/512/sae.pt`, `your_weights/512/dit.pt`,
`your_weights/512/latent_stats.pt`, and `your_configs/512/{sae,dit}.yaml`.
Use the downloaded YAMLs for these checkpoints, rather than the generic bundled
`dinov2_hf64.yaml` examples. Keep the released latent statistics for evaluating
the released DiT; the statistics-computation section is for training a new DiT.
The evaluator prefers `ema` if it is present in the DiT checkpoint.

DINOv2, ImageNet validation images (for PSNR/rFID), and the official FID reference
NPZs (for gFID) are separate downloads described below. You do not need the DINO
discriminator checkpoint for evaluation; it is needed for SAE GAN training.

### Download DINOv2 (with registers, base)

The DINOv2 examples use
[`facebook/dinov2-with-registers-base`](https://huggingface.co/facebook/dinov2-with-registers-base).
Download the complete Hugging Face model directory (including `config.json` and
weights), not just an isolated weight file:

```bash
python -m pip install huggingface_hub
python - <<'PYTHON'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="facebook/dinov2-with-registers-base",
    local_dir="your_pretrained/dinov2-with-registers-base",
)
PYTHON
```

In **each** SAE YAML used for training or evaluation, set:

```yaml
encoder:
  dinov2_model_name: your_pretrained/dinov2-with-registers-base
```

Keep the rest of the checkpoint's original `encoder` configuration unchanged.
The plain `facebook/dinov2-base` model is a different variant; do not substitute it
for a checkpoint trained with registers. You may also use the Hugging Face model
ID directly for automatic download, but an explicit local directory is convenient
for offline GPU nodes. See the [Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/guides/download).

### Where to put configs, weights, and statistics

All `your_xxxx` paths below are placeholders. Replace them with your own paths,
or create the shown layout under the repository root:

```text
your_pretrained/
  dinov2-with-registers-base/     # Downloaded backbone: config.json + weights
your_weights/
  256/
    sae.pt                       # Selected trained SAE checkpoint
    dit.pt                       # Matching trained DiT checkpoint (prefer EMA)
    latent_stats.pt              # Statistics used to TRAIN this DiT
  512/
    sae.pt
    dit.pt
    latent_stats.pt
your_configs/
  256/sae.yaml                    # Resolved SAE architecture/config
  256/dit.yaml                    # Resolved DiT architecture/config
  512/sae.yaml
  512/dit.yaml
your_fid_refs/
  VIRTUAL_imagenet256_labeled.npz
  VIRTUAL_imagenet512.npz
your_data/imagenet/
  train/<class_id>/*.JPEG
  val/<class_id>/*.JPEG
your_eval_outputs/               # Use a fresh subdirectory for each run
```

SAE/DiT checkpoints and their latent statistics are hosted on Hugging Face,
separately from this code repository. Use the download commands above or your own
trained artifacts; renaming a checkpoint to `sae.pt` or `dit.pt` does not change
its format. Copy the corresponding resolved YAMLs into `your_configs/<resolution>/`.
The bundled `dinov2_hf64.yaml` files are architecture examples, not universal
configs for arbitrary released checkpoints. In particular, a 512px checkpoint
needs its own matching config; changing only `data.image_size` is insufficient.

| Asset or setting | Where it is selected |
| --- | --- |
| DINOv2 model directory | SAE YAML: `encoder.dinov2_model_name` |
| SAE architecture | `eval_sae.py --config`; `eval_gfid.py --sae-config` |
| SAE checkpoint | `eval_sae.py --ckpt`; `eval_gfid.py --sae-ckpt` |
| DiT architecture/checkpoint | `eval_gfid.py --config` / `--dit-ckpt` |
| Training latent statistics | `--latent-stats-path`; DiT YAML: `misc.latent_stats_path` |
| Normalization mode | `--per-channel-norm` must match training |
| ImageNet validation images | `eval_sae.py --data-path` |
| gFID reference NPZ | `eval_gfid.py --fid-ref-path` |

Before use, replace machine-specific paths in your copied YAMLs: `data.train_path`,
`data.val_path`, `encoder.*_model_name`, `model.hf_encoder_config_path`,
`checkpoint.sae_ckpt`, `misc.latent_stats_path`, output directories, and any enabled
logging settings. Keep the bundled HF config path relative to this repository
when appropriate. Run commands from the repository root.

Other encoders use the corresponding `encoder.*_model_name`; DINOv3 uses
`encoder.dinov3_model_dir`. Select the exact backbone used during training.

The DINOv2 SAE example enables LPIPS and a DINO discriminator. Supply the matching
DINO ViT-S/8 discriminator weights at `discriminator.dino_ckpt_path` (default:
`pretrained/dino_vit_small_patch8_224.pth`). For a reconstruction-only experiment,
set `discriminator.enabled: false` and `loss.recon_gan_weight: 0.0`; this changes the
training recipe. LPIPS and FID may download their feature-network weights on first
use. Standard cache environment variables are respected.

W&B is disabled in the sample YAMLs. Set `logging.wandb_project` (SAE) or
`misc.wandb_project` (DiT) to enable it after configuring your own account.

<a id="train-sae"></a>

## 1. Train the SAE

Edit `dc-sae/configs/sae/dinov2_hf64.yaml` for your backbone, data and GPU
memory. `data.batch_size` is per GPU. Start single-node training directly:

```bash
torchrun --standalone --nproc_per_node=1 dc-sae/train_sae.py \
  --config dc-sae/configs/sae/dinov2_hf64.yaml \
  --train-path your_data/imagenet/train \
  --val-path your_data/imagenet/val \
  --output-dir your_results/sae_256
```

Set `--nproc_per_node` to the number of GPUs you want to use. Checkpoints are saved
under `your_results/sae_256/checkpoints/step_<N>.pth`. SAE training may resume from
an existing output directory; use a new directory for a new experiment. Keep a
copy of the exact YAML used for training with your checkpoints. For evaluation,
place the selected checkpoint at `your_weights/256/sae.pt` and its matching config
at `your_configs/256/sae.yaml`, or replace these placeholder paths consistently.

<a id="reconstruction"></a>

## 2. Evaluate reconstruction PSNR and FID

**PSNR** compares each reconstructed image with its own reference image. Higher is
better. **rFID** compares Inception feature distributions of reconstructions and
the same reference images. Lower is better. rFID is not generation FID.

```bash
python dc-sae/eval_sae.py \
  --config your_configs/256/sae.yaml \
  --ckpt your_weights/256/sae.pt \
  --data-path your_data/imagenet/val \
  --output-dir eval_outputs/sae_val \
  --batch-size 16 --num-workers 4 --precision fp32
```

For 512px reconstruction, use `your_configs/512/sae.yaml` and
`your_weights/512/sae.pt` with a separate output directory. No DiT checkpoint or
latent-statistics file is needed for direct reconstruction.

This is a **single-process/single-GPU** evaluator; do not launch it with torchrun.
By default it evaluates every image exactly once, including the final partial
batch. Use `--max-images 1000` for a quick partial evaluation, or `--skip-fid` for
PSNR only. Every run requires a new/empty output directory.

Outputs:

- `metrics.json`: sample count, PSNR in dB, rFID, configuration and checkpoint paths.
- `reference/` and `reconstruction/`: matched, flat PNG directories for FID.

Protocol: convert to RGB, apply the project's ADM-style center crop at
`data.image_size`, convert to `[-1,1]`, reconstruct, clamp to `[-1,1]`, and map to
`[0,1]`. PSNR is the arithmetic mean of per-image `-10 log10(MSE)` values, with MSE
computed in `[0,1]` and floored at `1e-10`. FID uses 8-bit PNGs and
`pytorch-fid`'s 2,048-dimensional Inception features. Use the same crop, resolution,
sample count and feature backend when comparing results. Report subset scores
as subset scores, rather than full ImageNet validation results.

<a id="latent-stats"></a>

## 3. Compute latent statistics before DiT training

Use the exact SAE checkpoint/config that will be frozen during DiT training:

```bash
torchrun --standalone --nproc_per_node=1 dc-sae/compute_latent_stats.py \
  --config your_configs/256/sae.yaml \
  --ckpt your_weights/256/sae.pt \
  --data-path your_data/imagenet/train \
  --num-samples 50000 --batch-size 32 \
  --output-dir your_weights/256
```

The script saves `latent_stats.pt` and `latent_stats.yaml`. The DiT normalizer uses
the `mean`/`std` fields of the **post-fused-norm** latent; `pre_*` fields are also
saved for analysis. The example DiT YAML sets `misc.per_channel_norm: true` and
points to this `.pt` file. Recompute statistics if the SAE weights, resolution,
latent layout or branch settings change. Statistics estimation uses the training
transform, including random horizontal flips. For multi-GPU execution, choose a
sample count divisible by the world size to avoid DistributedSampler padding.

<a id="train-dit"></a>

## 4. Train the DiT

Review `dc-sae/configs/sae/dinov2_hf64.yaml` and
`dc-sae/configs/dit/dinov2_hf64.yaml` together. The bundled example uses 256px
inputs, 832 latent channels (768 semantic + 64 HF), and a 16×16 token grid.
Use configs that match your own SAE checkpoint and latent layout.

```bash
torchrun --standalone --nproc_per_node=1 dc-sae/train_dit.py \
  --config dc-sae/configs/dit/dinov2_hf64.yaml \
  --sae-config dc-sae/configs/sae/dinov2_hf64.yaml \
  --sae-ckpt your_weights/256/sae.pt \
  --data-path your_data/imagenet/train \
  --results-dir your_results/dit_256 \
  --latent-stats-path your_weights/256/latent_stats.pt \
  --per-channel-norm --precision bf16 \
  --fid-ref-path your_fid_refs/VIRTUAL_imagenet256_labeled.npz
```

This example enables periodic FID using the official 256px reference from the
next section. Add `--skip-fid` to disable it. For a 512px model, supply its matching
SAE/DiT configs, weights, latent statistics and 512px reference file.
`training.global_batch_size` in the DiT YAML must be divisible by
`world_size * training.grad_accum_steps`.
Resume explicitly with `--ckpt your_weights/dit_resume.pt`, or use `--auto-resume`
to select a checkpoint from the output directory. Preserve the exact SAE/DiT
configs and latent statistics alongside the trained checkpoint.

### Multi-node training

Launch the same command on each node using your scheduler or shell. For two nodes
with four GPUs each, the first node runs:

```bash
torchrun --nnodes=2 --nproc_per_node=4 \
  --node_rank=0 --master_addr=your_master_host --master_port=29500 \
  dc-sae/train_sae.py \
  --config dc-sae/configs/sae/dinov2_hf64.yaml \
  --train-path your_data/imagenet/train \
  --val-path your_data/imagenet/val \
  --output-dir your_results/sae_256
```

On the second node use `--node_rank=1`; keep `--nnodes`, `--master_addr` and
`--master_port` the same. Replace `your_master_host` with the first node's reachable
address. Do not use `--standalone` for this multi-node command. All nodes need the
same environment, code, accessible input files and shared output directory.
For DiT, use the same torchrun prefix with `dc-sae/train_dit.py` and its arguments
shown above. Select GPU counts, networking and job submission for your own server.

<a id="gfid"></a>

## 5. Evaluate generation FID (gFID)

Generation FID compares **new class-conditional samples** from DiT + SAE decoder
against a real-image reference distribution. It does not compare paired
reconstructions. Two matching checkpoints are required: SAE and DiT.

### Download both official ImageNet reference files

Download the [OpenAI reference batches](https://github.com/openai/guided-diffusion/tree/main/evaluations)
before running gFID. Keep both files in `your_fid_refs/`; each evaluation uses
**only the file matching its output resolution**:

| Generated resolution | Reference file |
| --- | --- |
| 256 × 256 | [VIRTUAL_imagenet256_labeled.npz](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz) |
| 512 × 512 | [VIRTUAL_imagenet512.npz](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz) |

```bash
mkdir -p your_fid_refs
curl -fL --retry 3 -C - \
  -o your_fid_refs/VIRTUAL_imagenet256_labeled.npz \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
curl -fL --retry 3 -C - \
  -o your_fid_refs/VIRTUAL_imagenet512.npz \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz
```

These files contain precomputed reference statistics. `eval_gfid.py` reads
`mu[2048]` and `sigma[2048,2048]` directly; do not extract images or recompute their
statistics from ImageNet validation. A sample-only NPZ containing `arr_0` without
`mu`/`sigma` is not accepted. You can check the downloads without loading image arrays:

```bash
python - <<'PYTHON'
import numpy as np
for name in ("VIRTUAL_imagenet256_labeled.npz", "VIRTUAL_imagenet512.npz"):
    with np.load("your_fid_refs/" + name, allow_pickle=False) as ref:
        assert ref["mu"].shape == (2048,)
        assert ref["sigma"].shape == (2048, 2048)
    print("OK:", name)
PYTHON
```

The evaluator uses `pytorch-fid` for generated-image features; this is not a claim
of exact numerical equivalence to the original OpenAI TensorFlow evaluator.
Compare scores using the same feature backend, reference file, and sampling
settings. For historical training comparisons, use the exact historical reference
NPZ: switching a past 512px run from a 256px reference to the proper 512px reference
changes the evaluation protocol and requires reevaluation of the baseline.

For a deliberately different validation-reference protocol, you can use
`prepare_fid_reference.py`, but report that protocol separately. PSNR/rFID uses
validation images via `eval_sae.py` and does **not** use these gFID NPZs.

### Generate and evaluate 50,000 images at 256px

```bash
torchrun --standalone --nproc_per_node=1 dc-sae/eval_gfid.py \
  --config your_configs/256/dit.yaml \
  --sae-config your_configs/256/sae.yaml \
  --sae-ckpt your_weights/256/sae.pt \
  --dit-ckpt your_weights/256/dit.pt \
  --latent-stats-path your_weights/256/latent_stats.pt \
  --per-channel-norm \
  --fid-ref-path your_fid_refs/VIRTUAL_imagenet256_labeled.npz \
  --samples-per-class 50 --batch-size 32 --sample-steps 50 \
  --seed 42 --output-dir your_eval_outputs/gfid_256_run1
```

### Generate and evaluate 50,000 images at 512px

```bash
torchrun --standalone --nproc_per_node=1 dc-sae/eval_gfid.py \
  --config your_configs/512/dit.yaml \
  --sae-config your_configs/512/sae.yaml \
  --sae-ckpt your_weights/512/sae.pt \
  --dit-ckpt your_weights/512/dit.pt \
  --latent-stats-path your_weights/512/latent_stats.pt \
  --per-channel-norm \
  --fid-ref-path your_fid_refs/VIRTUAL_imagenet512.npz \
  --samples-per-class 50 --batch-size 8 --sample-steps 50 \
  --seed 42 --output-dir your_eval_outputs/gfid_512_run1
```

Both examples assume **per-channel normalization**. Use the exact statistics from
DiT training, even if evaluating a different compatible SAE decoder checkpoint.
Do not silently recompute or substitute another SAE checkpoint's statistics. If
your model was trained with scalar normalization, remove `--per-channel-norm` and
set `misc.per_channel_norm: false` in its DiT YAML.

For periodic training evaluation, pass the matching file to
`train_dit.py --fid-ref-path` and omit `--skip-fid`.

For 1,000 classes,
50 samples per class gives exactly 50,000 images. Classes are partitioned across
ranks; all ranks need the same shared output directory. Use a new output directory
for each run. Rank 0 computes FID after generation; failure to compute FID causes
a nonzero exit rather than a silently successful empty result.

- Default: **no CFG**. Add `--use-cfg --cfg-scale 1.5` for a guided run and report it
  separately. The inherited sampler applies guidance over linear-time `[0, 0.9]`.
- Sampling uses the project's x-prediction Euler update and dimension-dependent
  time shift. With `zero_hf_mode` or `hf_zero_then_joint_mode`, the shift uses
  semantic channels, matching training (including after the joint phase begins).
  `--sample-steps` controls evaluation; the evaluator does not use
  arbitrary `sampler`/`transport` YAML settings to switch algorithms.
- EMA weights are preferred when the DiT checkpoint contains `ema`; otherwise
  `model` or a raw state dictionary is loaded. Architecture weights load strictly.
- Use the same latent statistics and normalization mode as training. Keep HF
  enabled for the standard sample; use `--mask-hf --hf-mask-mode zero` only for a
  matching zero-HF model or an explicitly reported ablation.
- Output: `samples_no_mask/` and `metrics_no_mask.txt` (suffixes change for CFG/HF
  masking). The metric called `FID` in that file is **gFID** in this protocol.
- Optional `torch-fidelity` adds IS; `--ref-images-path` adds precision/recall.
  These optional metrics do not replace the `pytorch-fid` gFID score.

The seed is offset by rank. Changing GPU count, batch size, or precision may alter
the generated sample set, so report those settings along with CFG, sample steps,
resolution, number of samples, reference split and checkpoint.
`metrics.json` also records the evaluation arguments, sample count, latent-statistics
path, normalization mode, effective `time_shift`, `weight_source` (`ema`, `model`,
or `raw`) and `checkpoint_step`. Check `weight_source: "ema"` when comparing an EMA
training score.

## Validation

```bash
python -m unittest discover -s tests -v
```

Tests cover imports, training entrypoint help, strict checkpoint loading, PSNR
aggregation, latent normalization, and a complete CPU reconstruction-evaluation
run using a locally generated tiny DINOv2 model. No pretrained downloads are
needed for these tests. Full GPU training and benchmark PSNR/FID/gFID require
real model weights and datasets and are not claimed by these checks.

## License and attribution

The original MIT `LICENSE` and embedded third-party copyright headers are retained.
See `THIRD_PARTY_NOTICES.md`. Pretrained models and datasets have their own licenses
and access conditions. This package contains source code, not pretrained assets.
