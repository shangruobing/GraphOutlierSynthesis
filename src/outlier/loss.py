import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from icecream import ic

from src.outlier.energy import energy_propagation


def compute_loss(dataset_id: Data, dataset_ood: Data, encoder, classifier, criterion, device, args):
    """
    Compute the loss for in-distribution and out-of-distribution datasets.
    loss = supervised_learning_loss + energy_regularization_loss + classifier_loss
    Args:
        dataset_id: The in-distribution dataset
        dataset_ood: The out-of-distribution dataset
        encoder: The GNN encoder
        classifier: The penultimate embeddings classifier
        criterion: The loss function
        device: The device to run the computation
        args: The command line arguments

    Returns: The value of loss function.

    """
    loss = torch.tensor(data=0, dtype=torch.float, device=device)

    train_idx, train_ood_idx = dataset_id.train_mask, dataset_ood.node_idx

    # 获取GNN对ID数据的输出
    logits_id, penultimate_id = encoder(dataset_id.x, dataset_id.edge_index)
    logits_id, penultimate_id = logits_id[train_idx], penultimate_id[train_idx]

    # 使用GNN的ID输出计算loss
    predict_id = F.log_softmax(logits_id, dim=1)
    supervised_learning_loss = criterion(predict_id, dataset_id.y[train_idx].squeeze(1))
    loss += supervised_learning_loss

    # 获取GNN对OOD数据的输出
    logits_ood, penultimate_ood = encoder(dataset_ood.x, dataset_ood.edge_index)
    logits_ood, penultimate_ood = logits_ood[train_ood_idx], penultimate_ood[train_ood_idx]

    if args.use_energy:
        # 计算GNN输出的ID和OOD的能量分数
        # parser.add_argument('--T', type=float, default=1.0, help='temperature for Softmax')
        T = 1.0
        energy_id = - T * torch.logsumexp(logits_id / T, dim=-1)
        energy_ood = - T * torch.logsumexp(logits_ood / T, dim=-1)

        if args.use_energy_propagation:
            # 完成能量的传播
            # parser.add_argument('--num_prop_layers', type=int, default=2, help='number of layers for energy belief propagation')
            num_prop_layers = 2
            # parser.add_argument('--alpha', type=float, default=0.5, help='weight for residual connection in propagation')
            alpha = 0.5
            energy_id = energy_propagation(energy_id, dataset_id.edge_index, train_idx, num_prop_layers=num_prop_layers, alpha=alpha)
            energy_ood = energy_propagation(energy_ood, dataset_ood.edge_index, train_ood_idx, num_prop_layers=num_prop_layers, alpha=alpha)

        energy_id, energy_ood = trim_to_same_length(energy_id, energy_ood)

        # 计算能量的正则化损失
        # parser.add_argument('--upper_bound_id', type=float, default=-5, help='upper bound for in-distribution energy')
        upper_bound_id = -5
        # parser.add_argument('--lower_bound_ood', type=float, default=-1, help='lower bound for out-of-distribution energy')
        lower_bound_ood = -1
        # parser.add_argument('--lamda', type=float, default=1.0, help='weight for regularization')
        lamda = 1
        energy_regularization_loss = torch.mean(
            F.relu(energy_id - upper_bound_id) ** 2
            +
            F.relu(lower_bound_ood - energy_ood) ** 2
        )
        loss += lamda * energy_regularization_loss

        if args.use_classifier:
            # 将ID数据输入分类器
            classifier_id = classifier(
                penultimate_id,
                dataset_id.x,
                dataset_id.edge_index,
                train_idx
            )

            # 将OOD数据输入分类器
            classifier_ood = classifier(
                penultimate_ood,
                dataset_ood.x,
                dataset_ood.edge_index,
                train_ood_idx
            )

            if args.use_energy_filter:
                # 使用能量分数过滤分类器输出
                classifier_id, classifier_ood = filter_by_energy(classifier_id, classifier_ood, energy_id, energy_ood)

            # 构造分类器输出和标签
            min_length = min(len(classifier_id), len(classifier_ood))
            classifier_output = torch.cat([classifier_id[:min_length], classifier_ood[:min_length]])
            classifier_label = torch.cat([
                torch.ones(min_length, device=device),
                torch.zeros(min_length, device=device)
            ])

            classifier_loss = nn.BCELoss()(classifier_output, classifier_label)
            loss += classifier_loss

    return loss


def filter_by_energy(classifier_id, classifier_ood, energy_id, energy_ood, id_threshold=-5, ood_threshold=-5):
    """
    Filter the classifier output by energy scores.
    Args:
        classifier_id:
        classifier_ood:
        energy_id:
        energy_ood:
        id_threshold:
        ood_threshold:

    Returns:

    """
    filtered_classifier_id_index = torch.nonzero(energy_id < id_threshold).squeeze()
    filtered_classifier_ood_index = torch.nonzero(energy_ood > ood_threshold).squeeze()
    debug = False
    if debug:
        ic(energy_id.mean())
        ic(energy_ood.mean())
        ic(energy_id.shape)
        ic(filtered_classifier_id_index.shape)
        ic(energy_ood.shape)
        ic(filtered_classifier_ood_index.shape)
    return classifier_id[filtered_classifier_id_index], classifier_ood[filtered_classifier_ood_index]


def trim_to_same_length(energy_id: torch.Tensor, energy_ood: torch.Tensor):
    if energy_id.shape[0] != energy_ood.shape[0]:
        min_length = min(energy_id.shape[0], energy_ood.shape[0])
        energy_id = energy_id[:min_length]
        energy_ood = energy_ood[:min_length]
    return energy_id, energy_ood
