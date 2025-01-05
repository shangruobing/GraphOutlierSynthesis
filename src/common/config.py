import sys
from os.path import dirname, abspath
from pathlib import Path

BASE_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(BASE_DIR)

from src.common.utils import get_now_date

# The root path of the project
ROOT_PATH = Path(dirname(abspath(__file__))).parent.parent

# The path of the dataset
DATASET_PATH = ROOT_PATH / "dataset"

# The path of the result
FOLDER_PATH = ROOT_PATH / f"result/{get_now_date()}"
