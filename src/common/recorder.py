import os

import pandas as pd

from src.common.parse import Arguments
from src.common.config import FOLDER_PATH
from src.common.utils import get_now_datetime


class Recorder:
    @staticmethod
    def insert_row(
            args: Arguments,
            model: str,
            epoch_info: str,
            auroc: str,
            aupr: str,
            fpr: str,
            accuracy: str,
            score: str
    ) -> None:
        """
        将LLM的回答等信息存入运行结果表格
        Args:
            args: argparse,
            model: Model,
            epoch_info: str,
            auroc: str,
            aupr: str,
            fpr: str,
            accuracy: str,
            score: str,
        """
        new_row = {
            "args": args.string,
            "method": args.method,
            "backbone": args.backbone,
            "dataset": args.dataset,
            "ood_type": args.ood_type,
            "epochs": args.epochs,
            "use_energy": args.use_energy,
            "use_energy_propagation": args.use_energy_propagation,
            "use_classifier": args.use_classifier,
            "use_energy_filter": args.use_energy_filter,
            "synthesis_ood": args.synthesis_ood,
            "cov_mat": args.cov_mat,
            "sampling_ratio": args.sampling_ratio,
            "boundary_ratio": args.boundary_ratio,
            "boundary_sampling_ratio": args.boundary_sampling_ratio,
            "k": args.k,
            "model": model,
            "epoch_info": epoch_info,
            "time": get_now_datetime(),
            "AUROC": auroc,
            "ACCURACY": accuracy,
            "AUPR": aupr,
            "FPR": fpr,
            "SCORE": score,
        }
        FILE_PATH = FOLDER_PATH / f"{args.dataset}.csv"
        if not os.path.exists(FOLDER_PATH):
            os.mkdir(FOLDER_PATH)
        if os.path.exists(FILE_PATH):
            df = pd.read_csv(FILE_PATH, encoding="UTF-8-SIG")
        else:
            print(f"The {FILE_PATH} does not exist, a new file has been created.")
            df = pd.DataFrame(columns=list(new_row.keys()))
        df.loc[len(df)] = new_row
        df.to_csv(FILE_PATH, index=False, encoding="UTF-8-SIG")
