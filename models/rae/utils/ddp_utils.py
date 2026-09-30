"""
DDP/FSDP 分布式训练工具函数

包含:
- DDP/FSDP 初始化和清理
- 模型包装/解包工具
- EMA 更新
- Logger 创建
- 梯度控制

统一了 train_vae, eval_vae, projects/rae 中的重复代码
"""

import os
import logging
from collections import OrderedDict
from typing import Optional, Union, Tuple

from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 尝试导入 FSDP（可能在某些环境中不可用）
try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
        StateDictType,
        FullStateDictConfig,
    )
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    HAS_FSDP = True
except ImportError:
    HAS_FSDP = False
    FSDP = None


# ==========================================
#           DDP/FSDP 初始化
# ==========================================

def setup_ddp(backend: str = "nccl") -> Tuple[int, int, int]:
    """
    初始化分布式训练环境。
    
    Args:
        backend: 分布式后端，默认 "nccl"
    
    Returns:
        Tuple of (rank, local_rank, world_size)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # 标准 DDP 环境变量已设置（如 torchrun 启动）
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        print(f"[DDP] Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")
        return rank, local_rank, world_size
    else:
        # 单 GPU 运行的 fallback
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
        torch.cuda.set_device(0)
        print("[DDP] Running in single GPU mode.")
        return 0, 0, 1


def cleanup_ddp():
    """清理分布式进程组。"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """判断是否是主进程（rank 0）。"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """获取当前进程的 rank。"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """获取总进程数。"""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier():
    """同步所有进程。"""
    if dist.is_initialized():
        dist.barrier()


# ==========================================
#           模型包装/解包工具
# ==========================================

def requires_grad(model: nn.Module, flag: bool = True):
    """
    设置模型所有参数的 requires_grad 标志。
    
    Args:
        model: PyTorch 模型
        flag: 是否需要梯度
    """
    for p in model.parameters():
        p.requires_grad = flag


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    获取 DDP 或 FSDP 包装模型的内部原始模型。
    
    Args:
        model: 可能被 DDP/FSDP 包装的模型
    
    Returns:
        原始未包装的模型
    """
    if HAS_FSDP and isinstance(model, FSDP):
        return model.module
    elif isinstance(model, DDP):
        return model.module
    elif hasattr(model, "module"):
        return model.module
    return model


def get_model_state_dict(model: nn.Module) -> dict:
    """
    获取模型状态字典，兼容 DDP 和 FSDP。

    对于 FSDP，使用 full_state_dict 来收集所有分片。

    Args:
        model: 可能被 DDP/FSDP 包装的模型

    Returns:
        模型状态字典
    """
    if HAS_FSDP and isinstance(model, FSDP):
        # FSDP 需要特殊处理来收集完整的状态字典
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            return model.state_dict()
    elif isinstance(model, DDP):
        return model.module.state_dict()
    elif hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def get_optimizer_state_dict(
    model: Optional[nn.Module],
    optimizer: Optional[torch.optim.Optimizer],
) -> Optional[dict]:
    """
    获取 optimizer 状态字典，兼容普通模型、DDP 和 FSDP。

    对于 FSDP，所有 rank 都必须调用；只有 rank 0 会拿到完整状态字典。
    """
    if optimizer is None:
        return None
    if HAS_FSDP and model is not None and isinstance(model, FSDP):
        return FSDP.full_optim_state_dict(model, optimizer, rank0_only=True)
    return optimizer.state_dict()


def wrap_model_ddp(
    model: nn.Module,
    device_id: int,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = False,
) -> DDP:
    """
    使用 DDP 包装模型。
    
    Args:
        model: 要包装的模型
        device_id: GPU 设备 ID
        find_unused_parameters: 是否查找未使用的参数
        gradient_as_bucket_view: 是否使用 bucket view 优化
    
    Returns:
        DDP 包装后的模型
    """
    return DDP(
        model,
        device_ids=[device_id],
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
    )


def wrap_model_fsdp(
    model: nn.Module,
    device_id: int,
    use_bf16: bool = False,
    sharding_strategy: str = "FULL_SHARD",
    use_orig_params: bool = True,
    ignored_modules: Optional[list[nn.Module]] = None,
    auto_wrap_policy: Optional["ModuleWrapPolicy"] = None,
) -> "FSDP":
    """
    使用 FSDP 包装模型。
    
    Args:
        model: 要包装的模型
        device_id: GPU 设备 ID
        use_bf16: 是否使用 bf16 混合精度
        sharding_strategy: FSDP 分片策略
        use_orig_params: 是否使用原始参数（允许访问原始参数，方便 optimizer）
    
    Returns:
        FSDP 包装后的模型
    """
    if not HAS_FSDP:
        raise ImportError("FSDP not available in this PyTorch version.")
    
    fsdp_mixed_precision = None
    if use_bf16:
        fsdp_mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    
    # 解析 sharding_strategy
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    strategy = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)
    
    return FSDP(
        model,
        sharding_strategy=strategy,
        mixed_precision=fsdp_mixed_precision,
        device_id=device_id,
        use_orig_params=use_orig_params,
        ignored_modules=ignored_modules,
        auto_wrap_policy=auto_wrap_policy,
    )


# ==========================================
#           EMA 更新
# ==========================================

@torch.no_grad()
def update_ema(
    ema_model: nn.Module,
    model: nn.Module,
    decay: float = 0.9999,
    use_fsdp: bool = False,
):
    """
    更新 EMA 模型参数。
    
    支持 EMA 在 CPU 上而 model 在 GPU 上的情况。
    
    Args:
        ema_model: EMA 模型
        model: 当前训练模型（可能是 DDP/FSDP 包装的）
        decay: EMA decay rate
        use_fsdp: 是否使用 FSDP（需要使用 summon_full_params）
    """
    if use_fsdp and HAS_FSDP:
        # FSDP 需要使用 summon_full_params 来获取完整参数
        with FSDP.summon_full_params(model, writeback=False, recurse=True):
            model_inner = unwrap_model(model)
            _update_ema_params(ema_model, model_inner, decay)
    else:
        # DDP 或普通模型
        model_inner = unwrap_model(model)
        _update_ema_params(ema_model, model_inner, decay)


def _update_ema_params(ema_model: nn.Module, model: nn.Module, decay: float):
    """EMA 参数更新的内部实现。"""
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    ema_device = next(ema_model.parameters()).device
    
    for name, param in model_params.items():
        if name in ema_params:
            # 如果 EMA 在 CPU 上，需要先把 model 参数复制到 CPU
            param_data = param.data.to(ema_device) if param.device != ema_device else param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)


# ==========================================
#           梯度工具
# ==========================================


def clip_grad_norm(model: nn.Module, max_norm: float):
    """裁剪梯度范数，兼容 DDP 和 FSDP。"""
    if HAS_FSDP and isinstance(model, FSDP):
        return model.clip_grad_norm_(max_norm)
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def get_grad_norm(model: nn.Module):
    """获取梯度范数，兼容 DDP 和 FSDP。"""
    return clip_grad_norm(model, float("inf"))


# ==========================================
#           Logger 创建
# ==========================================

def create_logger(
    logging_dir: Optional[str] = None,
    rank: int = 0,
    name: str = "train",
    log_level: int = logging.INFO,
) -> logging.Logger:
    """
    创建 logger，只在 rank 0 时输出到文件和控制台。
    
    Args:
        logging_dir: 日志文件目录。如果为 None，只输出到控制台。
        rank: 当前进程的 rank
        name: logger 名称
        log_level: 日志级别
    
    Returns:
        配置好的 logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 清除已有的 handlers
    logger.handlers.clear()
    
    if rank == 0:
        # Rank 0: 输出到控制台和文件
        formatter = logging.Formatter(
            '[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台 handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # 文件 handler（如果提供了目录）
        if logging_dir is not None:
            os.makedirs(logging_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(logging_dir, "log.txt"), mode='a')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    else:
        # 其他 rank: 使用 NullHandler
        logger.addHandler(logging.NullHandler())
    
    return logger


# ==========================================
#           分布式聚合工具
# ==========================================

def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    对 tensor 进行 all_reduce 求平均。
    
    Args:
        tensor: 要聚合的张量
    
    Returns:
        聚合后的张量（所有 rank 的平均值）
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / dist.get_world_size()
    return tensor


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    对 tensor 进行 all_reduce 求和。
    
    Args:
        tensor: 要聚合的张量
    
    Returns:
        聚合后的张量（所有 rank 的和）
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    从 src rank 广播 tensor 到所有 rank。
    
    Args:
        tensor: 要广播的张量
        src: 源 rank
    
    Returns:
        广播后的张量
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    
    dist.broadcast(tensor, src=src)
    return tensor
