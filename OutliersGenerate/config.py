"""
全局配置文件
"""

import os
from pathlib import Path
from .utils import get_now_date
import sys
from os.path import dirname, abspath

BASE_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(BASE_DIR)

"""项目根目录"""
ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent

"""数据集路径"""

"""运行结果所在目录"""
FOLDER_PATH = ROOT_PATH / f"result/{get_now_date()}/"

"""运行结果文件路径"""
RUNNING_RESULT_PATH = FOLDER_PATH / "running_result.csv"

"""模板文件路径"""
TEMPLATE_PATH = ROOT_PATH / "template/"
