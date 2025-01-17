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

    def log(self, epoch: int, train_loss: float, auroc: float, aupr: float, fpr: float, accuracy: float, test_score: float, valid_loss: float):
        self.results.append((auroc, aupr, fpr, accuracy, test_score, valid_loss))
        info = (
            f'Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}, '
            f'AUROC: {100 * auroc:.2f}%, AUPR: {100 * aupr:.2f}%, FPR95: {100 * fpr:.2f}%, Accuracy: {100 * accuracy:.2f}%, '
            f'Test Score: {100 * test_score:.2f}%'
        )
        self.add_epoch_info(info)
        print(info)

    def add_epoch_info(self, info):
        self.epoch_info += info + '\n'

    def get_statistics(self, method: str) -> Metrics:
        result = 100 * torch.tensor(self.results)
        auroc, aupr, fpr, accuracy, test_score, valid_loss = result.T
        valid_loss = valid_loss / 100
        if method == "val_loss":
            min_index = valid_loss.argmin().item()
        elif method == "acc_val":
            min_index = accuracy.argmax().item()
        else:
            raise NotImplementedError
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
        print(
            f'Epoch: {min_index}, Loss(↓): {valid_loss[min_index]:.4f}, '
            f'AUROC(↑): {auroc_val:.2f}%, AUPR(↑): {aupr_val:.2f}%, FPR95(↓): {fpr_val:.2f}%, Accuracy(↑): {acc_val:.2f}%, '
            f'Test Score(↑): {score_val:.2f}%'
        )
        return Metrics(
            auroc=round(auroc_val, 2),
            aupr=round(aupr_val, 2),
            fpr=round(fpr_val, 2),
            accuracy=round(acc_val, 2),
            score=round(score_val, 2),
        )
