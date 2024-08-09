"""
Global configuration file
"""

import os
import sys
from pathlib import Path
from os.path import dirname, abspath

from src.common.utils import get_now_date

BASE_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(BASE_DIR)

# The root path of the project
ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent

# The path of the dataset
DATASET_PATH = ROOT_PATH / "dataset"

# The path of the result
FOLDER_PATH = ROOT_PATH / f"result/{get_now_date()}"
