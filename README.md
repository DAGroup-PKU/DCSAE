<div align="center">

# [NIPS2026🔥] DC-SAE: Deep Compression Semantic Autoencoder for Faster Diffusion Convergence

Xu Huang<sup>1*</sup> · Ye Huang<sup>1*</sup> · Zijun Liao<sup>1*</sup> · Yuwei Niu<sup>1</sup> · Xiaojie Li<br>
Menghan Zhou<sup>2</sup> · De Wen Soh<sup>2</sup> · Xiaotong Li<sup>1</sup> · Daquan Zhou<sup>1†</sup>

<sup>1</sup> Peking University &nbsp; <sup>2</sup> Singapore University of Technology and Design

<sup>*</sup> Equal contribution &nbsp; <sup>†</sup> Corresponding author

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](docs/installation-and-usage.md#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-training%20%26%20inference-EE4C2C?logo=pytorch&logoColor=white)](docs/installation-and-usage.md)
[![Hugging Face Models](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/DAGroup-PKU/DCSAE)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**English** · [简体中文](README-zn.md) · [Installation & Usage](docs/installation-and-usage.md)

</div>

![DC-SAE teaser: reconstruction, generation quality and throughput](assets/teaser.jpg)

## 🎬 Demo

https://github.com/user-attachments/assets/b29a4cd4-d103-4f7c-a890-2cdde307ef36

**[▶ Watch the demo](https://github.com/user-attachments/assets/b29a4cd4-d103-4f7c-a890-2cdde307ef36)** · [MP4](https://github.com/DAGroup-PKU/DCSAE/raw/refs/heads/main/assets/demo.mp4) · [English subtitles](assets/demo.vtt)

## Overview

**dc-sae** brings image reconstruction and latent-space generation into one training and evaluation pipeline. A pretrained vision encoder provides semantic features, an HF branch complements them, and a decoder reconstructs the image. DiT learns the resulting latent distribution for class-conditional generation.

The release includes SAE and DiT training, latent-statistics estimation, reconstruction **PSNR / rFID**, and generation **gFID** evaluation at **256px and 512px**, using configurations matched to each checkpoint.

| Component | Role |
| :--- | :--- |
| **Semantic + HF encoding** | Combine pretrained visual features with a complementary high-frequency branch. |
| **Reconstruction decoder** | Decode latents into images, with optional spatial demerger support. |
| **Latent DiT** | Train and sample a class-conditional generative model with the corresponding latent normalization. |
| **Evaluation** | Measure reconstruction PSNR/rFID and generation FID with explicit reference files and sampling settings. |

![dc-sae architecture: semantic and pixel encoding, spatial demerger, reconstruction and latent diffusion](assets/pipeline.png)

## Installation & Usage

```bash
git clone -b main --single-branch https://github.com/DAGroup-PKU/DCSAE.git
cd DCSAE
```

See the **[Installation & Usage guide](docs/installation-and-usage.md)** for dependencies, model downloads, weight placement, and complete commands. Run all commands from the repository root.

| Get started | Guide |
| :--- | :--- |
| Install dependencies | [Environment setup](docs/installation-and-usage.md#installation) |
| Download DINOv2 and prepare checkpoints | [Data & pretrained models](docs/installation-and-usage.md#assets) |
| Train the autoencoder | [SAE training](docs/installation-and-usage.md#train-sae) |
| Evaluate reconstruction quality | [PSNR & rFID](docs/installation-and-usage.md#reconstruction) |
| Prepare latents and train DiT | [Latent statistics](docs/installation-and-usage.md#latent-stats) · [DiT training](docs/installation-and-usage.md#train-dit) |
| Evaluate generation quality | [256px / 512px gFID](docs/installation-and-usage.md#gfid) |

> Pretrained backbones, trained SAE/DiT checkpoints, datasets, and FID references are downloaded or supplied separately. The guide includes the official ImageNet reference links and explains how to match checkpoints, configs, and latent statistics.

## Repository Structure

```text
dc-sae/       SAE / DiT training, evaluation, model modules and example configs
models/       DiT / DDT models and shared utilities
data/         ImageNet WebDataset support
train_vae/    GAN components for SAE training
docs/         Installation and usage guides in English and Chinese
tests/        Offline CPU checks
```

Training uses standard **torchrun** for single-node or multi-node execution. The bundled DINOv2 + HF64 configs provide an end-to-end architecture example; other checkpoints require their matching configs.

## License & Acknowledgements

Source code is distributed under the [MIT License](LICENSE). See [Third-party notices](THIRD_PARTY_NOTICES.md) for retained dependencies and their licenses. Pretrained models and datasets remain subject to their own terms.

Implementation checks and their scope are recorded in [VALIDATION.md](VALIDATION.md).
