import random
from datetime import datetime
from argparse import Namespace
import torch
import numpy as np


def get_device(args: Namespace):
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    return device


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_now_datetime() -> str:
    """
    获取今天的日期
    Returns:
        2023/10/01 12:30:30格式的日期时间
    """
    return datetime.now().strftime("%Y%m%d %H%M%S")


def get_now_date() -> str:
    """
    获取今天的日期
    Returns:
        2023/10/01格式的日期
    """
    return datetime.now().strftime("%Y%m%d")
