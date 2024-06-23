import random
from datetime import datetime
import torch
import numpy as np

from GNNSafe.parse import Arguments


def get_device(args: Arguments) -> torch.device:
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
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
    torch.backends.cudnn.deterministic = True


def get_now_datetime() -> str:
    """
    get now datetime
    Returns:
        2023-10-01 12:30:30
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_now_date() -> str:
    """
    get now date
    Returns:
        20231001
    """
    return datetime.now().strftime("%Y%m%d")
