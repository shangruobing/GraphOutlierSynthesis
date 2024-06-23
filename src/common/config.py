"""
全局配置文件
"""

import os
import sys
from pathlib import Path
from os.path import dirname, abspath

from src.common.utils import get_now_date

BASE_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(BASE_DIR)

"""项目根目录"""
ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent

"""数据集路径"""
DATASET_PATH = ROOT_PATH / "dataset"

"""运行结果所在目录"""
FOLDER_PATH = ROOT_PATH / f"result/{get_now_date()}"
