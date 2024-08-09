from dataclasses import dataclass

import torch

__all__ = ["DetectLogger"]


@dataclass
class Metrics:
    auroc: float
    aupr: float
    fpr: float
    accuracy: float
    score: float


class DetectLogger:
    """
    logger for ood detection task, reporting test auroc/aupr/fpr95 for ood detection
    """

    def __init__(self):
        self.results = []
        self.epoch_info = ""

    def add_result(self, result):
        """
        auroc, aupr, fpr, accuracy, test_score, valid_loss
        Args:
            result:

        Returns:

        """
        self.results.append(result)

    def add_epoch_info(self, info):
        self.epoch_info += info + '\n'

    def get_statistics(self) -> Metrics:
        result = 100 * torch.tensor(self.results)
        auroc, aupr, fpr, accuracy, test_score, valid_loss = result.T
        min_index = valid_loss.argmin().item()
        auroc_val = auroc[min_index].item()
        aupr_val = aupr[min_index].item()
        fpr_val = fpr[min_index].item()
        acc_val = accuracy[min_index].item()
        score_val = test_score[min_index].item()
        print(f'Choose Epoch: {min_index}')
        print(f'OOD Test Detect AUROC ↑: {auroc_val:.2f}')
        print(f'OOD Test Detect AUPR  ↑: {aupr_val:.2f}')
        print(f'OOD Test Detect FPR95 ↓: {fpr_val:.2f}')
        print(f'OOD Test Detect ACCU  ↑: {acc_val:.2f}')
        print(f'IND Test Accuracy     ↑: {score_val:.2f}')
        return Metrics(
            auroc=round(auroc_val, 2),
            aupr=round(aupr_val, 2),
            fpr=round(fpr_val, 2),
            accuracy=round(acc_val, 2),
            score=round(score_val, 2),
        )
