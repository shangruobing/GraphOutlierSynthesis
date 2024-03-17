import os
from argparse import Namespace
from pprint import pformat

import pandas as pd

from OutliersGenerate.config import FOLDER_PATH, RUNNING_RESULT_PATH
from OutliersGenerate.utils import get_now_datetime


def insert_row(
        args: Namespace,
        model: str,
        epoch_info: str,
        AUROC: str,
        AUPR: str,
        FPR: str,
        SCORE: str
) -> None:
    """
    将LLM的回答等信息存入运行结果表格
    Args:
        args: argparse,
        model: Model,
        epoch_info: str,
        AUROC: str,
        AUPR: str,
        FPR: str,
        SCORE: str,
    """
    FILE_PATH = FOLDER_PATH / f"{args.dataset}.csv"
    if not os.path.exists(FOLDER_PATH):
        os.mkdir(FOLDER_PATH)
    if os.path.exists(FILE_PATH):
        df = pd.read_csv(FILE_PATH, encoding="UTF-8-SIG")
    else:
        print(f"The {FILE_PATH} does not exist, an {FILE_PATH} file has been created.")
        df = pd.DataFrame(columns=[
            'args',
            "method",
            'generate_ood',
            "backbone",
            "dataset",
            "ood_type",
            "mode",
            "epochs",
            'model',
            'epoch_info',
            'time',
            'AUROC',
            'AUPR',
            'FPR',
            'SCORE'
        ])
    new_row = {
        "args": pformat(vars(args)),
        "method": args.method,
        "backbone": args.backbone,
        "dataset": args.dataset,
        "ood_type": args.ood_type,
        "mode": args.mode,
        "epochs": args.epochs,
        "generate_ood": args.generate_ood,
        "model": model,
        "epoch_info": epoch_info,
        "time": get_now_datetime(),
        "AUROC": AUROC,
        "AUPR": AUPR,
        "FPR": FPR,
        "SCORE": SCORE,
    }
    df.loc[len(df)] = new_row
    df.to_csv(FILE_PATH, index=False, encoding="UTF-8-SIG")
