"""
评估指标工具函数

包含:
- PSNR 计算
- LPIPS 损失
- SSIM 计算 (可选)
- FID 计算辅助

统一了 train_vae, eval_vae, projects/rae 中的重复代码
"""

from typing import Tuple, Optional, Union
import torch
import torch.nn as nn

# 尝试导入 lpips
try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("Warning: lpips not installed. LPIPS loss will not be available.")

# 尝试导入 pytorch-fid
try:
    from pytorch_fid import fid_score
    HAS_FID = True
except ImportError:
    HAS_FID = False


# ==========================================
#           PSNR 计算
# ==========================================

def calculate_psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: Tuple[float, float] = (-1.0, 1.0),
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    计算两张图像之间的 PSNR。
    
    Args:
        img1: 第一张图像 [B, C, H, W] 或 [C, H, W]
        img2: 第二张图像，形状与 img1 相同
        data_range: 图像值范围，默认 (-1, 1)
        eps: 防止除零的小值
    
    Returns:
        PSNR 值（标量或每个 batch 样本的 PSNR）
    """
    # 转换到 [0, 1] 范围
    low, high = data_range
    img1 = (img1 - low) / (high - low)
    img2 = (img2 - low) / (high - low)
    
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    
    # 计算 MSE
    mse = torch.mean((img1 - img2) ** 2)
    
    if mse <= eps:
        return torch.tensor(100.0, device=img1.device)
    
    # PSNR = 20 * log10(MAX / sqrt(MSE))
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def calculate_batch_psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: Tuple[float, float] = (-1.0, 1.0),
    eps: float = 1e-10,
) -> Tuple[float, int]:
    """
    批量计算 PSNR，返回总和和数量（用于分布式聚合）。
    
    Args:
        img1: 第一张图像 [B, C, H, W]
        img2: 第二张图像 [B, C, H, W]
        data_range: 图像值范围
        eps: 防止除零的小值
    
    Returns:
        Tuple of (psnr_sum, count)
    """
    # 转换到 [0, 1] 范围
    low, high = data_range
    img1 = (img1 - low) / (high - low)
    img2 = (img2 - low) / (high - low)
    
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    
    # 计算每个样本的 MSE
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])  # [B]
    mse = torch.clamp(mse, min=eps)
    
    # 计算每个样本的 PSNR
    psnr = 10 * torch.log10(1.0 / mse)  # [B]
    
    return psnr.sum().item(), psnr.shape[0]


def calculate_psnr_per_sample(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: Tuple[float, float] = (-1.0, 1.0),
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    计算每个样本的 PSNR。
    
    Args:
        img1: 第一张图像 [B, C, H, W]
        img2: 第二张图像 [B, C, H, W]
        data_range: 图像值范围
        eps: 防止除零的小值
    
    Returns:
        每个样本的 PSNR [B]
    """
    # 转换到 [0, 1] 范围
    low, high = data_range
    img1 = (img1 - low) / (high - low)
    img2 = (img2 - low) / (high - low)
    
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    
    # 计算每个样本的 MSE
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])  # [B]
    mse = torch.clamp(mse, min=eps)
    
    # 计算每个样本的 PSNR
    psnr = 10 * torch.log10(1.0 / mse)
    
    return psnr


# ==========================================
#           LPIPS 损失
# ==========================================

class LPIPSLoss(nn.Module):
    """
    LPIPS 感知损失。
    
    封装 lpips 库，输入期望在 [-1, 1] 范围。
    
    Args:
        net: 网络类型 ('vgg', 'alex', 'squeeze')
        device: 设备
        use_lpips_weighting: 是否使用 LPIPS 的学习权重
        use_dropout: 是否使用 dropout
    
    Example:
        >>> lpips_loss = LPIPSLoss(device='cuda')
        >>> loss = lpips_loss(img1, img2)
    """
    
    def __init__(
        self,
        net: str = 'vgg',
        device: Union[str, torch.device] = 'cuda',
        use_lpips_weighting: bool = False,
        use_dropout: bool = False,
    ):
        super().__init__()
        
        if not HAS_LPIPS:
            raise ImportError(
                "lpips not installed. Install with: pip install lpips"
            )
        
        self.lpips = lpips.LPIPS(
            net=net,
            lpips=use_lpips_weighting,
            use_dropout=use_dropout,
        ).to(device)
        self.lpips.eval()
        
        # 冻结参数
        for p in self.lpips.parameters():
            p.requires_grad = False
    
    def forward(
        self,
        x: torch.Tensor,
        rec: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        计算 LPIPS 损失。
        
        Args:
            x: 原始图像 [B, C, H, W]，范围 [-1, 1]
            rec: 重建图像 [B, C, H, W]，范围 [-1, 1]
            normalize: 是否内部归一化（如果输入在 [0, 1] 则设为 True）
        
        Returns:
            LPIPS 损失值
        """
        return self.lpips(x, rec, normalize=normalize).mean()
    
    def forward_per_sample(
        self,
        x: torch.Tensor,
        rec: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        计算每个样本的 LPIPS 损失。
        
        Args:
            x: 原始图像 [B, C, H, W]
            rec: 重建图像 [B, C, H, W]
            normalize: 是否内部归一化
        
        Returns:
            每个样本的 LPIPS 损失 [B]
        """
        # LPIPS 返回 [B, 1, 1, 1]
        loss = self.lpips(x, rec, normalize=normalize)
        return loss.view(-1)


# ==========================================
#           L1/L2 损失
# ==========================================

def l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    计算 L1 损失。
    
    Args:
        pred: 预测值
        target: 目标值
        reduction: 归约方式 ('none', 'mean', 'sum')
    
    Returns:
        L1 损失
    """
    return torch.nn.functional.l1_loss(pred, target, reduction=reduction)


def l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    计算 L2 (MSE) 损失。
    
    Args:
        pred: 预测值
        target: 目标值
        reduction: 归约方式 ('none', 'mean', 'sum')
    
    Returns:
        L2 损失
    """
    return torch.nn.functional.mse_loss(pred, target, reduction=reduction)


# ==========================================
#           组合重建损失
# ==========================================

class ReconstructionLoss(nn.Module):
    """
    组合重建损失（L1 + LPIPS）。
    
    常用于 VAE 训练。
    
    Args:
        l1_weight: L1 损失权重
        lpips_weight: LPIPS 损失权重
        lpips_net: LPIPS 网络类型
        device: 设备
    
    Example:
        >>> recon_loss = ReconstructionLoss(l1_weight=1.0, lpips_weight=0.5, device='cuda')
        >>> loss, loss_dict = recon_loss(pred, target)
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        lpips_weight: float = 0.5,
        lpips_net: str = 'vgg',
        device: Union[str, torch.device] = 'cuda',
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        
        self.lpips_loss = None
        if lpips_weight > 0 and HAS_LPIPS:
            self.lpips_loss = LPIPSLoss(net=lpips_net, device=device)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算组合重建损失。
        
        Args:
            pred: 预测图像 [B, C, H, W]，范围 [-1, 1]
            target: 目标图像 [B, C, H, W]，范围 [-1, 1]
        
        Returns:
            Tuple of (total_loss, loss_dict)
            loss_dict 包含各项损失的值
        """
        loss_dict = {}
        total_loss = 0.0
        
        # L1 损失
        if self.l1_weight > 0:
            l1 = l1_loss(pred, target)
            loss_dict['l1'] = l1.item()
            total_loss = total_loss + self.l1_weight * l1
        
        # LPIPS 损失
        if self.lpips_loss is not None and self.lpips_weight > 0:
            lpips_val = self.lpips_loss(target, pred)
            loss_dict['lpips'] = lpips_val.item()
            total_loss = total_loss + self.lpips_weight * lpips_val
        
        loss_dict['total'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return total_loss, loss_dict


# ==========================================
#           FID 计算辅助
# ==========================================

def calculate_fid(
    path1: str,
    path2: str,
    batch_size: int = 50,
    device: Union[str, torch.device] = 'cuda',
    dims: int = 2048,
    num_workers: int = 8,
) -> float:
    """
    计算两个图像目录之间的 FID 分数。
    
    Args:
        path1: 第一个图像目录路径
        path2: 第二个图像目录路径
        batch_size: 批次大小
        device: 设备
        dims: Inception 特征维度
        num_workers: 数据加载器 workers 数量
    
    Returns:
        FID 分数
    """
    if not HAS_FID:
        raise ImportError(
            "pytorch-fid not installed. Install with: pip install pytorch-fid"
        )
    
    return fid_score.calculate_fid_given_paths(
        paths=[path1, path2],
        batch_size=batch_size,
        device=device,
        dims=dims,
        num_workers=num_workers,
    )


def is_fid_available() -> bool:
    """检查 FID 计算是否可用。"""
    return HAS_FID


def is_lpips_available() -> bool:
    """检查 LPIPS 损失是否可用。"""
    return HAS_LPIPS
