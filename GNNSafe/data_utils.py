from typing import Tuple

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import torch
import numpy as np
from torch_sparse import SparseTensor
import torch.nn.functional as F
from baselines import ODIN, Mahalanobis
from torch import BoolTensor


def rand_splits(num_nodes, train_ratio=0.5, test_ratio=0.25) -> Tuple[BoolTensor, BoolTensor, BoolTensor]:
    """
    Randomly split the nodes into train, val, and test sets
    Args:
        num_nodes:
        train_ratio:
        test_ratio:

    Returns:

    """

    train_size = int(train_ratio * num_nodes)
    test_size = int(test_ratio * num_nodes)

    # 创建索引
    indices = torch.arange(num_nodes)

    # 随机打乱索引
    indices = indices[torch.randperm(num_nodes)]

    # 划分数据集
    train_indices = indices[:train_size]
    test_indices = indices[train_size:train_size + test_size]
    val_indices = indices[train_size + test_size:]

    # 创建布尔索引
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    val_mask[val_indices] = True

    return train_mask, val_mask, test_mask


def to_sparse_tensor(edge_index, edge_feat, num_nodes):
    """ converts the edge_index into SparseTensor
    """
    num_edges = edge_index.size(1)

    (row, col), N, E = edge_index, num_nodes, num_edges
    perm = (col * N + row).argsort()
    row, col = row[perm], col[perm]

    value = edge_feat[perm]
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(N, N), is_sorted=True)

    adj_t.storage.rowptr()
    adj_t.storage.csr2csc()

    return adj_t


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    """
    Calculate the FPR95
    Args:
        y_true:
        y_score:
        recall_level:
        pos_label:

    Returns:

    """
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))
    if np.array_equal(classes, [1]):
        return thresholds[cutoff]  # return threshold

    return fps[cutoff] / (np.sum(np.logical_not(y_true))), thresholds[cutoff]


def get_measures(_pos, _neg, recall_level=0.95):
    """
    The OOD detection performance
    Args:
        _pos:
        _neg:
        recall_level:

    Returns:

    """
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1
    auroc = roc_auc_score(labels, examples)
    aupr = average_precision_score(labels, examples)
    fpr, threshould = fpr_and_fdr_at_recall(labels, examples, recall_level)
    return auroc, aupr, fpr, threshould


def eval_acc(y_true, y_pred):
    """
    The model prediction performance
    Args:
        y_true:
        y_pred:

    Returns:

    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy().reshape(-1)
    return accuracy_score(y_true=y_true, y_pred=y_pred)


def eval_rocauc(y_true, y_pred):
    """
    Model binary classification performance
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py
    Args:
        y_true:
        y_pred:

    Returns:
    """
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)


@torch.no_grad()
def evaluate_classify(model, dataset, eval_func, criterion, args, device):
    model.eval()

    train_idx, valid_idx, test_idx = dataset.train_mask, dataset.val_mask, dataset.test_mask
    y = dataset.y
    out = model(dataset, device).cpu()

    train_score = eval_func(y[train_idx], out[train_idx])
    valid_score = eval_func(y[valid_idx], out[valid_idx])
    test_score = eval_func(y[test_idx], out[test_idx])

    if args.method != 'GPN':
        if args.dataset in ('proteins', 'ppi'):
            valid_loss = criterion(out[valid_idx], y[valid_idx].to(torch.float))
        else:
            valid_out = F.log_softmax(out[valid_idx], dim=1)
            valid_loss = criterion(valid_out, y[valid_idx].squeeze(1))

        return train_score, valid_score, test_score, valid_loss
    else:
        return train_score, valid_score, test_score


def evaluate_detect(model, dataset_ind, dataset_ood, criterion, eval_func, args, device):
    """
    OOD detection performance
    Args:
        model:
        dataset_ind:
        dataset_ood:
        criterion:
        eval_func:
        args:
        device:

    Returns:

    """
    model.eval()

    if isinstance(model, Mahalanobis):
        test_ind_score = model.detect(dataset_ind, dataset_ind.train_mask, dataset_ind, dataset_ind.test_mask,
                                      device, args)
    elif isinstance(model, ODIN):
        test_ind_score = model.detect(dataset_ind, dataset_ind.test_mask, device, args).cpu()
    else:
        with torch.no_grad():
            test_ind_score = model.detect(dataset_ind, dataset_ind.test_mask, device, args).cpu()
    if isinstance(model, Mahalanobis):
        test_ood_score = model.detect(dataset_ind, dataset_ind.train_mask, dataset_ood, dataset_ood.node_idx,
                                      device, args).cpu()
    elif isinstance(model, ODIN):
        test_ood_score = model.detect(dataset_ood, dataset_ood.node_idx, device, args).cpu()
    else:
        with torch.no_grad():
            test_ood_score = model.detect(dataset_ood, dataset_ood.node_idx, device, args).cpu()

    auroc, aupr, fpr, _ = get_measures(test_ind_score, test_ood_score)
    result = [auroc] + [aupr] + [fpr]

    out = model(dataset_ind, device).cpu()
    test_idx = dataset_ind.test_mask
    # test_idx = dataset_ind.test_mask
    test_score = eval_func(dataset_ind.y[test_idx], out[test_idx])

    valid_idx = dataset_ind.val_mask
    if args.dataset in ('proteins', 'ppi'):
        valid_loss = criterion(out[valid_idx], dataset_ind.y[valid_idx].to(torch.float))
    else:
        valid_out = F.log_softmax(out[valid_idx], dim=1)
        valid_loss = criterion(valid_out, dataset_ind.y[valid_idx].squeeze(1))

    result += [test_score] + [valid_loss]
    return result
