"""WebDataset-based ImageNet dataloader for DiT training.

Provides ``ImageNetWDS``, an ``IterableDataset`` that reads from multi-tar
WebDataset shards, supporting both offline (pre-extracted features) and
online (encode-on-the-fly) modes.

Both modes yield ``(latent: Tensor[C, H, W], label: int)`` tuples.

Typical usage::

    # Offline – pre-extracted features
    dataset = ImageNetWDS("/data/imagenet_wds", mode="offline",
                          feat_key="feat_dc_sae_dinov2_hf256")
    loader = build_wds_dataloader(dataset, batch_size=128)

    # Online – encode on the fly
    from data.encoders import DCSAEEncoder
    enc = DCSAEEncoder(config, ckpt, device="cuda")
    dataset = ImageNetWDS("/data/imagenet_wds", mode="online", encoder=enc)
"""

from __future__ import annotations

import io
import json
import os
import random
import tarfile
from glob import glob
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms

from data.encoders import LatentEncoder


# -----------------------------------------------------------------------
#  Image transform (matches train_dit.py / generate_dc_sae_latent.py)
# -----------------------------------------------------------------------

def _center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def _default_image_transform(image_size: int):
    return transforms.Compose([
        transforms.Lambda(lambda img: _center_crop_arr(img, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),
    ])


# -----------------------------------------------------------------------
#  Core dataset
# -----------------------------------------------------------------------

class ImageNetWDS(IterableDataset):
    """IterableDataset over multi-tar ImageNet WebDataset shards.

    Args:
        wds_root:    Root directory containing ``image/``, ``label/``, and
                     ``feat_*/`` subdirectories of ``.tar`` files.
        mode:        ``"offline"`` to read pre-extracted features or
                     ``"online"`` to encode images on-the-fly.
        feat_key:    Subdirectory name for offline features (e.g.
                     ``"feat_dc_sae_dinov2_hf256"``).  Required when
                     ``mode="offline"``.
        encoder:     A ``LatentEncoder`` instance.  Required when
                     ``mode="online"``.
        image_size:  Input resolution for online encoding.
        shuffle:     Whether to shuffle shard order each epoch.
        seed:        Base random seed for deterministic shuffling.
    """

    def __init__(
        self,
        wds_root: str,
        mode: str = "offline",
        feat_key: Optional[str] = None,
        encoder: Optional[LatentEncoder] = None,
        image_size: int = 256,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        assert mode in ("offline", "online"), f"mode must be 'offline' or 'online', got {mode!r}"

        self.wds_root = wds_root
        self.mode = mode
        self.feat_key = feat_key
        self.encoder = encoder
        self.image_size = image_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        # Discover label shards (always needed)
        self._label_shards = sorted(glob(os.path.join(wds_root, "label", "*.tar")))
        if not self._label_shards:
            raise FileNotFoundError(f"No label tars found in {wds_root}/label/")

        if mode == "offline":
            if feat_key is None:
                raise ValueError("feat_key is required for offline mode")
            self._feat_shards = sorted(glob(os.path.join(wds_root, feat_key, "*.tar")))
            if not self._feat_shards:
                raise FileNotFoundError(f"No feature tars found in {wds_root}/{feat_key}/")
            if len(self._feat_shards) != len(self._label_shards):
                raise ValueError(
                    f"Shard count mismatch: {feat_key}/ has {len(self._feat_shards)}, "
                    f"label/ has {len(self._label_shards)}"
                )
            self._image_shards = None
        else:  # online
            if encoder is None:
                raise ValueError("encoder is required for online mode")
            self._image_shards = sorted(glob(os.path.join(wds_root, "image", "*.tar")))
            if not self._image_shards:
                raise FileNotFoundError(f"No image tars found in {wds_root}/image/")
            if len(self._image_shards) != len(self._label_shards):
                raise ValueError(
                    f"Shard count mismatch: image/ has {len(self._image_shards)}, "
                    f"label/ has {len(self._label_shards)}"
                )
            self._feat_shards = None
            self._transform = _default_image_transform(image_size)

        self._num_shards = len(self._label_shards)

        # Load wdinfo for total sample count
        wdinfo_path = os.path.join(wds_root, "wdinfo.json")
        if os.path.exists(wdinfo_path):
            with open(wdinfo_path) as f:
                info = json.load(f)
            self._total_samples = info.get("total_samples", self._num_shards * 1024)
        else:
            self._total_samples = self._num_shards * 1024  # estimate

    # -- epoch control ---------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shard shuffling (call before each epoch)."""
        self._epoch = epoch

    # -- length ----------------------------------------------------------

    def __len__(self) -> int:
        """Approximate total samples (from wdinfo or estimate)."""
        return self._total_samples

    # -- shard distribution ----------------------------------------------

    def _get_worker_shards(self) -> list[int]:
        """Distribute shard indices across DDP ranks and DataLoader workers."""
        indices = list(range(self._num_shards))

        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(indices)

        # DDP split
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank, world_size = 0, 1
        indices = indices[rank::world_size]

        # DataLoader worker split
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            indices = indices[worker_info.id :: worker_info.num_workers]

        return indices

    # -- iteration -------------------------------------------------------

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        for shard_idx in self._get_worker_shards():
            if self.mode == "offline":
                yield from self._iter_offline_shard(shard_idx)
            else:
                yield from self._iter_online_shard(shard_idx)

    def _read_labels(self, shard_idx: int) -> dict[str, int]:
        """Read all UID → label mappings from a label tar."""
        labels: dict[str, int] = {}
        with tarfile.open(self._label_shards[shard_idx], "r") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                uid = os.path.splitext(member.name)[0]
                data = tf.extractfile(member).read()
                labels[uid] = int(data.decode("utf-8").strip())
        return labels

    def _iter_offline_shard(self, shard_idx: int):
        """Yield (latent, label) from pre-extracted feature + label tars."""
        labels = self._read_labels(shard_idx)

        with tarfile.open(self._feat_shards[shard_idx], "r") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                uid = os.path.splitext(member.name)[0]
                data = tf.extractfile(member).read()
                arr = np.load(io.BytesIO(data))
                latent = torch.from_numpy(arr).float()  # [C, H, W]
                label = labels.get(uid, -1)
                yield latent, label

    def _iter_online_shard(self, shard_idx: int):
        """Read images, encode on-the-fly, yield (latent, label)."""
        labels = self._read_labels(shard_idx)

        # Read all images from the shard
        uids = []
        images = []
        with tarfile.open(self._image_shards[shard_idx], "r") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                uid = os.path.splitext(member.name)[0]
                data = tf.extractfile(member).read()
                img = Image.open(io.BytesIO(data)).convert("RGB")
                uids.append(uid)
                images.append(self._transform(img))

        if not uids:
            return

        # Batch encode
        img_batch = torch.stack(images)  # [N, 3, H, W]
        chunk_size = 32
        latents = []
        for i in range(0, len(uids), chunk_size):
            chunk = img_batch[i : i + chunk_size]
            # Determine encoder device from its internal state
            if hasattr(self.encoder, "_device"):
                chunk = chunk.to(self.encoder._device, non_blocking=True)
            lat = self.encoder.encode(chunk)
            latents.append(lat.cpu().float())

        all_latents = torch.cat(latents, dim=0)  # [N, C, h, w]

        for j, uid in enumerate(uids):
            label = labels.get(uid, -1)
            yield all_latents[j], label


# -----------------------------------------------------------------------
#  Convenience builder
# -----------------------------------------------------------------------

def build_wds_dataloader(
    wds_root: str,
    mode: str,
    feat_key: Optional[str] = None,
    encoder: Optional[LatentEncoder] = None,
    image_size: int = 256,
    batch_size: int = 128,
    num_workers: int = 4,
    shuffle: bool = True,
    seed: int = 42,
    epoch: int = 0,
) -> DataLoader:
    """Build a ``DataLoader`` wrapping ``ImageNetWDS``.

    Returns a standard ``DataLoader`` that yields
    ``(latent_batch, label_batch)`` tuples.
    """
    dataset = ImageNetWDS(
        wds_root=wds_root,
        mode=mode,
        feat_key=feat_key,
        encoder=encoder,
        image_size=image_size,
        shuffle=shuffle,
        seed=seed,
    )
    dataset.set_epoch(epoch)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
