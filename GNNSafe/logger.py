import torch


class DetectLogger:
    """
    logger for ood detection task, reporting test auroc/aupr/fpr95 for ood detection
    """

    def __init__(self):
        self.results = []

    def add_result(self, result):
        self.results.append(result)

    def get_statistics(self):
        result = 100 * torch.tensor(self.results)
        ood_result, test_score, valid_loss = result[:, :-2], result[:, -2], result[:, -1]
        score_val = test_score[valid_loss.argmin()].item()
        auroc_val = ood_result[valid_loss.argmin(), 0].item()
        aupr_val = ood_result[valid_loss.argmin(), 1].item()
        fpr_val = ood_result[valid_loss.argmin(), 2].item()
        print(f'OOD Test Final AUROC: {auroc_val:.2f}')
        print(f'OOD Test Final AUPR: {aupr_val:.2f}')
        print(f'OOD Test Final FPR: {fpr_val:.2f}')
        print(f'IND Test Score: {score_val:.2f}')
        return {
            "AUROC": round(score_val, 2),
            "AUPR": round(auroc_val, 2),
            "FPR": round(aupr_val, 2),
            "SCORE": round(fpr_val, 2),
        }


class ClassifyLogger:
    """
    logger for node classification task, reporting train/valid/test accuracy or roc, auc for classification
    """

    def __init__(self):
        self.results = []

    def add_result(self, result):
        self.results.append(result)

    def get_statistics(self):
        result = 100 * torch.tensor(self.results)
        train1 = result[:, 0].max().item()
        test1 = result[:, 2].max().item()
        valid = result[:, 1].max().item()
        train2 = result[result[:, 1].argmax(), 0].item()
        test2 = result[result[:, 1].argmax(), 2].item()
        print(f'Highest Train: {train1:.2f}')
        print(f'Highest Test: {test1:.2f}')
        print(f'Highest Valid: {valid:.2f}')
        print(f'  Final Train: {train2:.2f}')
        print(f'   Final Test: {test2:.2f}')
        return {
            "Highest Train": round(train1, 2),
            "Highest Test": round(test1, 2),
            "Highest Valid": round(valid, 2),
            "Final Train": round(train2, 2),
            "Final Test": round(test2, 2),
        }
