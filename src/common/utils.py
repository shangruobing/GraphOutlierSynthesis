import random
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch

__all__ = ["get_device", "fix_seed", "get_now_datetime", "get_now_date"]


def get_device(device: str, cpu=False) -> torch.device:
    if cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{device}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def get_now_datetime(timezone: str = "Asia/Shanghai") -> str:
    """
    get now datetime
    Returns:
        2023-10-01 12:30:30
    """
    return datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M:%S")


def get_now_date(timezone: str = "Asia/Shanghai") -> str:
    """
    get now date
    Returns:
        20231001
    """
    return datetime.now(ZoneInfo(timezone)).strftime("%Y%m%d")
