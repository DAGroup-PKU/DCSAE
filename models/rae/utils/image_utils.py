"""
图像预处理工具函数

包含:
- 中心裁剪
- 训练/验证数据增强 transforms
- 图像格式转换

统一了 train_vae, eval_vae, projects/rae 中的重复代码
"""

from typing import Tuple, Union, Callable
import numpy as np
from PIL import Image

import torch
from torchvision import transforms


# ==========================================
#           图像裁剪
# ==========================================

def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """
    ADM 风格的中心裁剪。
    
    先将图像缩小到目标尺寸的 2 倍以内，然后缩放到目标尺寸并中心裁剪。
    
    Args:
        pil_image: PIL 图像
        image_size: 目标尺寸
    
    Returns:
        裁剪后的 PIL 图像
    """
    # 逐步缩小到目标尺寸的 2 倍以内
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    
    # 缩放到目标尺寸
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    
    # 中心裁剪
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def random_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """
    随机裁剪图像。
    
    Args:
        pil_image: PIL 图像
        image_size: 目标尺寸
    
    Returns:
        随机裁剪后的 PIL 图像
    """
    # 逐步缩小到目标尺寸的 2 倍以内
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    
    # 缩放使得最小边等于目标尺寸
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    
    # 随机裁剪
    arr = np.array(pil_image)
    max_y = arr.shape[0] - image_size
    max_x = arr.shape[1] - image_size
    
    crop_y = np.random.randint(0, max_y + 1) if max_y > 0 else 0
    crop_x = np.random.randint(0, max_x + 1) if max_x > 0 else 0
    
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


# ==========================================
#           Transform 工厂函数
# ==========================================

def get_train_transform(
    image_size: int,
    augment: bool = True,
    center_crop: bool = True,
    normalize_range: Tuple[float, float] = (-1.0, 1.0),
) -> transforms.Compose:
    """
    获取训练数据增强 transform。
    
    Args:
        image_size: 目标图像尺寸
        augment: 是否应用数据增强（随机翻转等）
        center_crop: 是否使用中心裁剪（否则使用 RandomResizedCrop）
        normalize_range: 归一化范围，默认 (-1, 1)
    
    Returns:
        torchvision transforms 组合
    """
    transform_list = []
    
    if center_crop:
        # 使用自定义的中心裁剪
        transform_list.append(
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size))
        )
    else:
        # 使用随机缩放裁剪
        transform_list.append(
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0))
        )
    
    if augment:
        transform_list.append(transforms.RandomHorizontalFlip())
    
    transform_list.append(transforms.ToTensor())  # [0, 1]
    
    # 归一化到指定范围
    if normalize_range == (-1.0, 1.0):
        transform_list.append(transforms.Lambda(lambda t: t * 2.0 - 1.0))
    elif normalize_range != (0.0, 1.0):
        low, high = normalize_range
        transform_list.append(
            transforms.Lambda(lambda t: t * (high - low) + low)
        )
    
    return transforms.Compose(transform_list)


def get_val_transform(
    image_size: int,
    normalize_range: Tuple[float, float] = (-1.0, 1.0),
) -> transforms.Compose:
    """
    获取验证/测试 transform（无数据增强）。
    
    Args:
        image_size: 目标图像尺寸
        normalize_range: 归一化范围，默认 (-1, 1)
    
    Returns:
        torchvision transforms 组合
    """
    transform_list = [
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.ToTensor(),  # [0, 1]
    ]
    
    # 归一化到指定范围
    if normalize_range == (-1.0, 1.0):
        transform_list.append(transforms.Lambda(lambda t: t * 2.0 - 1.0))
    elif normalize_range != (0.0, 1.0):
        low, high = normalize_range
        transform_list.append(
            transforms.Lambda(lambda t: t * (high - low) + low)
        )
    
    return transforms.Compose(transform_list)


def get_augment_transform(
    image_size: int,
    use_random_resize_crop: bool = True,
    scale_range: Tuple[float, float] = (0.8, 1.0),
    normalize_range: Tuple[float, float] = (-1.0, 1.0),
) -> transforms.Compose:
    """
    获取带增强的训练 transform（用于 VAE 训练）。
    
    Args:
        image_size: 目标图像尺寸
        use_random_resize_crop: 是否使用随机缩放裁剪
        scale_range: 随机缩放范围
        normalize_range: 归一化范围
    
    Returns:
        torchvision transforms 组合
    """
    transform_list = []
    
    if use_random_resize_crop:
        transform_list.append(
            transforms.RandomResizedCrop(image_size, scale=scale_range)
        )
    else:
        transform_list.append(
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size))
        )
    
    transform_list.extend([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    # 归一化
    if normalize_range == (-1.0, 1.0):
        transform_list.append(transforms.Lambda(lambda t: t * 2.0 - 1.0))
    elif normalize_range != (0.0, 1.0):
        low, high = normalize_range
        transform_list.append(
            transforms.Lambda(lambda t: t * (high - low) + low)
        )
    
    return transforms.Compose(transform_list)


# ==========================================
#           图像格式转换
# ==========================================

def tensor_to_pil(
    tensor: torch.Tensor,
    normalize_range: Tuple[float, float] = (-1.0, 1.0),
) -> Image.Image:
    """
    将 tensor 转换为 PIL 图像。
    
    Args:
        tensor: 形状为 [C, H, W] 或 [H, W, C] 的张量
        normalize_range: 输入 tensor 的值范围
    
    Returns:
        PIL 图像
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # 取 batch 中第一个
    
    if tensor.shape[0] in [1, 3, 4]:  # [C, H, W] 格式
        tensor = tensor.permute(1, 2, 0)  # -> [H, W, C]
    
    # 反归一化到 [0, 1]
    low, high = normalize_range
    tensor = (tensor - low) / (high - low)
    tensor = tensor.clamp(0, 1)
    
    # 转换为 uint8
    arr = (tensor.cpu().numpy() * 255).astype(np.uint8)
    
    if arr.shape[-1] == 1:
        arr = arr.squeeze(-1)  # 灰度图
    
    return Image.fromarray(arr)


def pil_to_tensor(
    pil_image: Image.Image,
    normalize_range: Tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """
    将 PIL 图像转换为 tensor。
    
    Args:
        pil_image: PIL 图像
        normalize_range: 输出 tensor 的值范围
    
    Returns:
        形状为 [C, H, W] 的张量
    """
    arr = np.array(pil_image).astype(np.float32) / 255.0  # [0, 1]
    
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]  # [H, W] -> [H, W, 1]
    
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
    
    # 归一化到指定范围
    low, high = normalize_range
    tensor = tensor * (high - low) + low
    
    return tensor


def denormalize_tensor(
    tensor: torch.Tensor,
    normalize_range: Tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """
    将归一化的 tensor 转换到 [0, 1] 范围。
    
    Args:
        tensor: 归一化的张量
        normalize_range: 输入 tensor 的值范围
    
    Returns:
        [0, 1] 范围的张量
    """
    low, high = normalize_range
    tensor = (tensor - low) / (high - low)
    return tensor.clamp(0, 1)


def normalize_tensor(
    tensor: torch.Tensor,
    target_range: Tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """
    将 [0, 1] 范围的 tensor 归一化到指定范围。
    
    Args:
        tensor: [0, 1] 范围的张量
        target_range: 目标值范围
    
    Returns:
        归一化后的张量
    """
    low, high = target_range
    return tensor * (high - low) + low


# ==========================================
#           图像保存/加载辅助
# ==========================================

def save_tensor_as_image(
    tensor: torch.Tensor,
    path: str,
    normalize_range: Tuple[float, float] = (-1.0, 1.0),
):
    """
    将 tensor 保存为图像文件。
    
    Args:
        tensor: 形状为 [C, H, W] 的张量
        path: 保存路径
        normalize_range: 输入 tensor 的值范围
    """
    pil_image = tensor_to_pil(tensor, normalize_range)
    pil_image.save(path)


def load_image_as_tensor(
    path: str,
    image_size: int = None,
    normalize_range: Tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """
    加载图像文件为 tensor。
    
    Args:
        path: 图像路径
        image_size: 如果指定，将图像调整为此尺寸
        normalize_range: 输出 tensor 的值范围
    
    Returns:
        形状为 [C, H, W] 的张量
    """
    pil_image = Image.open(path).convert('RGB')
    
    if image_size is not None:
        pil_image = center_crop_arr(pil_image, image_size)
    
    return pil_to_tensor(pil_image, normalize_range)
