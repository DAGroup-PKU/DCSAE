# dc-sae

[English](README.md) | 简体中文

本项目是 dc-sae 图像重建与 latent 空间 DiT 流程的独立版本，包含 SAE 训练、latent 统计量估计、DiT 训练、重建 PSNR/rFID 评估和生成 FID（gFID）评估。

这是一个独立的新仓库，不包含原仓库的 Git 历史、实验报告、凭据、数据集或训练权重。

## 项目内容

```text
dc-sae/
  train_sae.py, train_dit.py    # 训练实现
  compute_latent_stats.py       # DiT 使用的 latent 均值与标准差
  eval_sae.py                   # 成对 PSNR 与重建 FID
  eval_gfid.py                  # DiT 采样与生成 FID
  prepare_fid_reference.py      # ImageFolder 转为 pytorch-fid 统计量
  model.py, modules/            # SAE、编码器、解码器、HF 分支及 demerger
  configs/                     # 通用示例配置
models/                        # DiT/DDT、解码器、DINOv3 和公共工具
train_vae/                     # SAE 训练所需的 GAN 组件
data/                          # DiT 的 ImageNet WebDataset 支持
tests/                         # 离线 CPU 检查
```

编码器实现包括 DINOv2、DINOv2 deep compression、DINOv3、SigLIP2、Qwen3-ViT（含 pooling / 可学习 merger 变体）和 InternVL3。随附的端到端示例使用 DINOv2 + HF64 及匹配的 DiT/DDT 配置；另有 Qwen3 HF256/512px SAE 架构示例，它与 DINOv2 的 DiT 配置不兼容。

## 环境安装

请使用 Python 3.10+（随附 CPU 检查使用 Python 3.12）。训练和分布式生成需要 Linux 与 CUDA GPU。创建并激活虚拟环境，先安装适合机器的 CUDA 版 `torch` / `torchvision`，然后安装以下依赖：

```bash
python -m pip install -r requirements.txt
# 可选：W&B 日志及 IS / precision / recall
python -m pip install -r requirements-optional.txt
```

所有命令均在**仓库根目录**、已激活的 Python 环境中执行。训练和分布式生成直接使用标准 `torchrun`，不依赖 tmux、SSH 自动化、特定 Conda 安装或集群启动器。代码和输出建议放在支持 POSIX 语义的文件系统中；对象存储挂载可能不支持训练需要的全部文件操作。

`requirements.txt` 定义依赖版本范围；`requirements-tested.txt` 记录本地导入和 CPU 验证所用的版本。这些检查不代表 CUDA 基准测试，也不代表所有编码器均完成了端到端验证。

## 数据与预训练模型

数据使用 ImageFolder 布局，验证集也需要按类别建立子目录：

```text
your_data/imagenet/train/<class_id>/*.JPEG
your_data/imagenet/val/<class_id>/*.JPEG
```

SAE 数据路径可通过 YAML 的 `data.train_path`、`data.val_path` 或命令行的 `--train-path`、`--val-path` 指定；DiT 数据路径通过 `--data-path` 指定。训练集与验证集的类别顺序必须一致。默认生成示例使用 ImageNet 的 1,000 个类别。本仓库不分发数据集、预训练权重或训练得到的权重。

### 下载 DINOv2（带 registers 的 base 版本）

DINOv2 示例使用 [`facebook/dinov2-with-registers-base`](https://huggingface.co/facebook/dinov2-with-registers-base)。请下载完整的 Hugging Face 模型目录，包括 `config.json` 和模型权重，而不只是单独一个权重文件：

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

在**每一份**用于训练或评估的 SAE YAML 中设置：

```yaml
encoder:
  dinov2_model_name: your_pretrained/dinov2-with-registers-base
```

保留该 checkpoint 原有的其余 `encoder` 配置。普通的 `facebook/dinov2-base` 是另一种模型，不能替代训练时使用的带 registers 版本。也可以直接填写 Hugging Face 模型 ID 以自动下载；离线 GPU 节点更适合使用明确的本地目录。下载方法参见 [Hugging Face 官方指南](https://huggingface.co/docs/huggingface_hub/guides/download)。

### 配置、权重与统计文件的存放位置

以下所有 `your_xxxx` 路径都是占位符，请替换成自己的路径，或在仓库根目录创建如下目录结构：

```text
your_pretrained/
  dinov2-with-registers-base/     # 下载的 backbone：config.json 与权重
your_weights/
  256/
    sae.pt                       # 选定的 SAE 训练权重
    dit.pt                       # 匹配的 DiT 训练权重（优先 EMA）
    latent_stats.pt              # 训练此 DiT 时使用的统计量
  512/
    sae.pt
    dit.pt
    latent_stats.pt
your_configs/
  256/sae.yaml                    # 实际训练使用的 SAE 架构配置
  256/dit.yaml                    # 实际训练使用的 DiT 架构配置
  512/sae.yaml
  512/dit.yaml
your_fid_refs/
  VIRTUAL_imagenet256_labeled.npz
  VIRTUAL_imagenet512.npz
your_data/imagenet/
  train/<class_id>/*.JPEG
  val/<class_id>/*.JPEG
your_eval_outputs/               # 每次运行使用新的子目录
```

仓库**不附带** SAE/DiT checkpoint 和 latent 统计文件，请使用自己的训练产物。把文件重命名为 `sae.pt` 或 `dit.pt` 不会改变其格式。将对应的 resolved YAML（已解析配置）复制到 `your_configs/<resolution>/`。随附的 `dinov2_hf64.yaml` 仅为架构示例，并非适用于任意权重的通用配置。512px checkpoint 必须配套对应配置，仅修改 `data.image_size` 并不足够。

| 资源或设置 | 指定位置 |
| --- | --- |
| DINOv2 模型目录 | SAE YAML: `encoder.dinov2_model_name` |
| SAE 架构 | `eval_sae.py --config`; `eval_gfid.py --sae-config` |
| SAE 权重 | `eval_sae.py --ckpt`; `eval_gfid.py --sae-ckpt` |
| DiT 架构/权重 | `eval_gfid.py --config` / `--dit-ckpt` |
| 训练 latent 统计量 | `--latent-stats-path`; DiT YAML: `misc.latent_stats_path` |
| 归一化模式 | `--per-channel-norm` 必须与训练一致 |
| ImageNet 验证图像 | `eval_sae.py --data-path` |
| gFID 参考 NPZ | `eval_gfid.py --fid-ref-path` |

使用前请替换所复制 YAML 中的机器相关路径：`data.train_path`、`data.val_path`、`encoder.*_model_name`、`model.hf_encoder_config_path`、`checkpoint.sae_ckpt`、`misc.latent_stats_path`、输出目录及已启用的日志设置。随附 HF 配置可保留为相对于仓库根目录的路径。命令应在仓库根目录运行。

其他编码器使用各自对应的 `encoder.*_model_name`；DINOv3 使用 `encoder.dinov3_model_dir`。务必选择训练时使用的同一个 backbone。

DINOv2 SAE 示例启用了 LPIPS 和 DINO 判别器。请在 `discriminator.dino_ckpt_path` 指定匹配的 DINO ViT-S/8 判别器权重，默认位置为 `pretrained/dino_vit_small_patch8_224.pth`。如果仅进行重建训练，可设置 `discriminator.enabled: false` 和 `loss.recon_gan_weight: 0.0`，但这会改变训练方案。LPIPS 和 FID 首次使用时可能下载特征网络权重，可通过标准缓存环境变量设置缓存位置。

示例 YAML 默认关闭 W&B。配置自己的账户后，可通过 SAE 的 `logging.wandb_project` 或 DiT 的 `misc.wandb_project` 启用。

## 1. 训练 SAE

根据实际 backbone、数据和显存修改 `dc-sae/configs/sae/dinov2_hf64.yaml`，其中 `data.batch_size` 为每张 GPU 的 batch size。单机直接运行：

```bash
torchrun --standalone --nproc_per_node=1 dc-sae/train_sae.py \
  --config dc-sae/configs/sae/dinov2_hf64.yaml \
  --train-path your_data/imagenet/train \
  --val-path your_data/imagenet/val \
  --output-dir your_results/sae_256
```

将 `--nproc_per_node` 改为要使用的 GPU 数量。checkpoint 保存在 `your_results/sae_256/checkpoints/step_<N>.pth`。SAE 训练可能从已有输出目录恢复，新实验请使用新目录。请随 checkpoint 保存实际使用的 YAML。评估示例将选定 checkpoint 放在 `your_weights/256/sae.pt`、配套配置放在 `your_configs/256/sae.yaml`，也可以统一替换为自己的路径。

## 2. 评估重建 PSNR 与 rFID

**PSNR** 对比每张重建图与对应原图，越高越好；**rFID** 对比重建图和同一批原图的 Inception 特征分布，越低越好。rFID 是重建指标，不是生成 FID。

```bash
python dc-sae/eval_sae.py \
  --config your_configs/256/sae.yaml \
  --ckpt your_weights/256/sae.pt \
  --data-path your_data/imagenet/val \
  --output-dir eval_outputs/sae_val \
  --batch-size 16 --num-workers 4 --precision fp32
```

评估 512px 重建时，请换用 `your_configs/512/sae.yaml` 和 `your_weights/512/sae.pt`，并使用独立输出目录。直接重建不需要 DiT checkpoint 或 latent 统计文件。

此评估器是**单进程、单 GPU** 程序，不要用 torchrun 启动。默认每张图恰好评估一次，包括最后不足一个 batch 的图像。快速检查可加 `--max-images 1000`，只计算 PSNR 可加 `--skip-fid`。每次运行都需要新的或空的输出目录。

输出文件如下：

- `metrics.json`：样本数、PSNR（dB）、rFID、配置与 checkpoint 路径。
- `reference/` 和 `reconstruction/`：一一对应、无类别子目录的 PNG 图像，用于计算 FID。

评估流程：转为 RGB，按 `data.image_size` 使用项目的 ADM 风格中心裁剪，映射到 `[-1,1]` 后重建，将结果截断至 `[-1,1]` 再映射到 `[0,1]`。PSNR 为逐图 `-10 log10(MSE)` 的算术平均，MSE 在 `[0,1]` 范围内计算，最小值限制为 `1e-10`。FID 使用 8 位 PNG 和 `pytorch-fid` 的 2,048 维 Inception 特征。比较结果时应统一裁剪、分辨率、样本数和特征提取后端；子集结果不能当作完整 ImageNet 验证集结果报告。

## 3. 在 DiT 训练前计算 latent 统计量

必须使用接下来 DiT 训练中冻结的同一份 SAE checkpoint 和配置：

```bash
torchrun --standalone --nproc_per_node=1 dc-sae/compute_latent_stats.py \
  --config your_configs/256/sae.yaml \
  --ckpt your_weights/256/sae.pt \
  --data-path your_data/imagenet/train \
  --num-samples 50000 --batch-size 32 \
  --output-dir your_weights/256
```

脚本保存 `latent_stats.pt` 和 `latent_stats.yaml`。DiT 归一化使用 **post-fused-norm** latent 的 `mean` / `std`；`pre_*` 字段供分析使用。示例 DiT YAML 设置 `misc.per_channel_norm: true` 并指向该 `.pt` 文件。为新训练配置更换 SAE 权重、分辨率、latent 布局或分支设置时，需重新计算统计量。统计过程使用包含随机水平翻转的训练变换。多 GPU 运行时，样本数应能被 GPU 进程数整除，以避免 DistributedSampler 补齐样本。

## 4. 训练 DiT

一起核对 `dc-sae/configs/sae/dinov2_hf64.yaml` 和 `dc-sae/configs/dit/dinov2_hf64.yaml`。随附示例使用 256px 输入、832 个 latent 通道（768 语义 + 64 HF）及 16×16 token 网格。实际训练时必须使用与 SAE checkpoint 和 latent 布局匹配的配置。

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

示例使用下一节的官方 256px 参考文件启用训练期 FID，添加 `--skip-fid` 可关闭。512px 模型需要换用对应的 SAE/DiT 配置、权重、latent 统计量和 512px 参考文件。DiT YAML 中的 `training.global_batch_size` 必须能被 `world_size * training.grad_accum_steps` 整除。

恢复训练可显式指定 `--ckpt your_weights/dit_resume.pt`，或使用 `--auto-resume` 从输出目录选择 checkpoint。请随训练权重保留实际使用的 SAE/DiT 配置和 latent 统计量。

### 多节点训练

使用自己的调度器或终端，在每个节点运行同一训练命令。以两个节点、每节点四张 GPU 为例，第一个节点运行：

```bash
torchrun --nnodes=2 --nproc_per_node=4 \
  --node_rank=0 --master_addr=your_master_host --master_port=29500 \
  dc-sae/train_sae.py \
  --config dc-sae/configs/sae/dinov2_hf64.yaml \
  --train-path your_data/imagenet/train \
  --val-path your_data/imagenet/val \
  --output-dir your_results/sae_256
```

第二个节点将 `--node_rank` 改为 `1`，保持 `--nnodes`、`--master_addr` 和 `--master_port` 相同。将 `your_master_host` 替换为第一个节点可访问的地址；多节点命令不要使用 `--standalone`。各节点需要相同环境、代码、可访问的输入文件及共享输出目录。DiT 训练使用相同的 torchrun 前缀，换成 `dc-sae/train_dit.py` 及上文参数。GPU 数量、网络设置和作业提交方式由用户按服务器配置选择。

## 5. 评估生成 FID（gFID）

gFID 将 DiT + SAE 解码器**新生成的类别条件图像**与真实图像参考分布进行比较，不进行成对重建对比。需要相互匹配的 SAE 和 DiT 两份 checkpoint。

### 下载两个分辨率的官方 ImageNet 参考文件

运行 gFID 前，请下载下方 OpenAI 官方参考文件，统一放在 `your_fid_refs/`。每次评估**只使用与生成图像分辨率对应的文件**：

| 生成分辨率 | 参考文件 |
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

这些 NPZ 包含预计算的参考统计量。`eval_gfid.py` 直接读取 `mu[2048]` 和 `sigma[2048,2048]`，不需要提取图像，也不要用 ImageNet 验证集重新计算并替换它们。只包含 `arr_0` 而没有 `mu` / `sigma` 的图像 NPZ 无法作为参考文件使用。下面的检查只读取统计量，不加载图像数组：

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

本评估器使用 `pytorch-fid` 提取生成图像特征，不保证与 OpenAI 原版 TensorFlow 评估器数值完全相同。比较分数时必须统一特征后端、参考文件和采样设置。若要复现历史训练分数，需要使用历史评估时的同一份 NPZ：如果以前的 512px 评估用了 256px 参考文件，改成正确的 512px 参考文件会改变评估协议，应重新评估对照结果。

如果明确要使用另一套基于验证集的参考分布，可运行 `prepare_fid_reference.py`，但应单独说明该评估协议。PSNR/rFID 通过 `eval_sae.py` 使用验证集图像，**不使用**这里的 gFID NPZ。

### 生成并评估 50,000 张 256px 图像

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

### 生成并评估 50,000 张 512px 图像

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

两个示例均假设模型采用**逐通道归一化**。即使换用了另一个兼容的 SAE 解码器，也必须使用 DiT 训练时的原始统计量，不能擅自重新计算或替换成另一个 SAE checkpoint 的统计量。若训练时使用标量归一化，请移除 `--per-channel-norm`，并将 DiT YAML 中的 `misc.per_channel_norm` 设为 `false`。

训练期定期评估时，给 `train_dit.py --fid-ref-path` 传入对应分辨率的文件，并且不要添加 `--skip-fid`。

ImageNet 有 1,000 个类别，每类生成 50 张即为 50,000 张。类别分配给不同 rank，所有 rank 必须使用同一个共享输出目录。每次运行使用新目录。生成结束后由 rank 0 计算 FID；若计算失败，程序会以非零状态退出。

- 默认**不启用 CFG**。需要引导采样时添加 `--use-cfg --cfg-scale 1.5` 并单独报告；引导应用在线性时间 `[0, 0.9]` 区间。
- 使用项目的 x-prediction Euler 更新及与维度相关的 time shift。启用 `zero_hf_mode` 或 `hf_zero_then_joint_mode` 时，以语义通道数计算 time shift，与训练保持一致，进入 joint 阶段后也不改变。步数由 `--sample-steps` 控制，不能通过任意 `sampler` / `transport` YAML 设置切换算法。
- checkpoint 含 `ema` 时优先加载 EMA，否则加载 `model` 或原始 state dict；架构权重严格匹配加载。
- 使用与训练相同的 latent 统计量和归一化方式。标准评估保留 HF；仅在匹配的 zero-HF 模型或明确报告的消融实验中使用 `--mask-hf --hf-mask-mode zero`。
- 输出为 `samples_no_mask/` 和 `metrics_no_mask.txt`，CFG/HF masking 会改变后缀。其中的 `FID` 即本协议下的 **gFID**。
- 可选的 `torch-fidelity` 提供 IS，`--ref-images-path` 提供 precision/recall；这些指标不替代 `pytorch-fid` 的 gFID 分数。

随机种子按 rank 偏移。GPU 数量、batch size 或精度变化可能改变生成样本，因此报告结果时请注明这些参数，以及 CFG、采样步数、分辨率、样本数、参考数据和 checkpoint。`metrics.json` 记录评估参数、样本数、latent 统计路径、归一化模式、实际 `time_shift`、`weight_source`（`ema` / `model` / `raw`）和 `checkpoint_step`。与训练期 EMA 分数比较时，请确认 `weight_source: "ema"`。

## 验证

```bash
python -m unittest discover -s tests -v
```

检查覆盖模块导入、训练入口的帮助命令、严格 checkpoint 加载、PSNR 汇总、latent 归一化，以及使用本地创建的微型 DINOv2 模型完成 CPU 重建评估。这些检查不需要下载预训练权重。完整 GPU 训练和基准 PSNR/FID/gFID 需要真实权重与数据集，不能由这些 CPU 检查替代。

## 许可证与致谢

保留原 MIT `LICENSE` 和第三方源码中的版权声明，详见 `THIRD_PARTY_NOTICES.md`。预训练模型和数据集各自适用其许可证与访问条件。本项目分发源代码，不附带预训练资源。