import torch
import torch.nn as nn
import torch.nn.functional as F
# from icecream import ic
from torch_geometric.data import Data
from OutliersGenerate.energy import energy_propagation


def compute_loss(dataset_ind: Data, dataset_ood: Data, encoder, classifier, criterion, device, args):
    """
    Compute the loss for in-distribution and out-of-distribution datasets.
    Args:
        dataset_ind: The in-distribution dataset
        dataset_ood: The out-of-distribution dataset
        encoder: The GNN encoder
        classifier: The penultimate embeddings classifier
        criterion: The loss function
        device: The device to run the computation
        args: The command line arguments

    Returns:

    """
    train_idx, train_ood_idx = dataset_ind.train_mask, dataset_ood.node_idx

    # 获取GNN对ID数据的输出
    logits_in, penultimate_in = encoder(dataset_ind.x, dataset_ind.edge_index)
    logits_in, penultimate_in = logits_in[train_idx], penultimate_in[train_idx]

    # 使用GNN的ID输出计算loss
    predict_in = F.log_softmax(logits_in, dim=1)
    loss = criterion(predict_in, dataset_ind.y[train_idx].squeeze(1))

    # 获取GNN对OOD数据的输出
    logits_out, penultimate_ood = encoder(dataset_ood.x, dataset_ood.edge_index)
    logits_out, penultimate_ood = logits_out[train_ood_idx], penultimate_ood[train_ood_idx]

    # 计算GNN输出的ID和OOD的能量分数
    # parser.add_argument('--T', type=float, default=1.0, help='temperature for Softmax')
    T = 1.0
    energy_in = - T * torch.logsumexp(logits_in / T, dim=-1)
    energy_out = - T * torch.logsumexp(logits_out / T, dim=-1)

    # 完成能量的传播
    # parser.add_argument('--K', type=int, default=2, help='number of layers for energy belief propagation')
    K = 2
    # parser.add_argument('--alpha', type=float, default=0.5, help='weight for residual connection in propagation')
    alpha = 0.5
    energy_in = energy_propagation(energy_in, dataset_ind.edge_index, train_idx, K, alpha)
    energy_out = energy_propagation(energy_out, dataset_ood.edge_index, train_ood_idx, K, alpha)
    energy_in, energy_out = trim_to_same_length(energy_in, energy_out)

    # 计算能量的正则化损失
    # parser.add_argument('--m_in', type=float, default=-5, help='upper bound for in-distribution energy')
    m_in = -5
    # parser.add_argument('--m_out', type=float, default=-1, help='lower bound for in-distribution energy')
    m_out = -1
    # parser.add_argument('--lamda', type=float, default=1.0, help='weight for regularization')
    lamda = 1
    energy_regularization_loss = torch.mean(F.relu(energy_in - m_in) ** 2 + F.relu(m_out - energy_out) ** 2)
    loss += lamda * energy_regularization_loss

    if args.use_classifier:
        # 将ID数据输入分类器
        classifier_in = classifier(
            penultimate_in,
            dataset_ind.x,
            dataset_ind.edge_index,
            train_idx
        )

        # 将OOD数据输入分类器
        classifier_ood = classifier(
            penultimate_ood,
            dataset_ood.x,
            dataset_ood.edge_index,
            train_ood_idx
        )

        # 使用能量分数过滤分类器输出
        classifier_in, classifier_ood = filter_by_energy(classifier_in, classifier_ood, energy_in, energy_out)

        # 构造分类器输出和标签
        min_length = min(len(classifier_in), len(classifier_ood))
        classifier_output = torch.cat([classifier_in[:min_length], classifier_ood[:min_length]])

        classifier_label = torch.cat([
            torch.ones(min_length, device=device),
            torch.zeros(min_length, device=device)
        ])

        classifier_loss = nn.BCELoss()(classifier_output, classifier_label)
        loss += classifier_loss

    return loss


def filter_by_energy(classifier_in, classifier_out, energy_in, energy_out, threshold=-5):
    """
    Filter the classifier output by energy scores.
    Args:
        classifier_in:
        classifier_out:
        energy_in:
        energy_out:
        threshold:

    Returns:

    """
    filtered_classifier_in_index = torch.nonzero(energy_in < threshold).squeeze()
    filtered_classifier_index = torch.nonzero(energy_out > threshold).squeeze()
    return classifier_in[filtered_classifier_in_index], classifier_out[filtered_classifier_index]


def trim_to_same_length(energy_in: torch.Tensor, energy_out: torch.Tensor):
    if energy_in.shape[0] != energy_out.shape[0]:
        min_length = min(energy_in.shape[0], energy_out.shape[0])
        energy_in = energy_in[:min_length]
        energy_out = energy_out[:min_length]
    return energy_in, energy_out
