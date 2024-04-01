from icecream import ic
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from OutliersGenerate.KNN import generate_outliers


def compute_loss(dataset_ind: Data, dataset_ood: Data, encoder, classifier, criterion, device, args):
    """
    Compute the loss for in-distribution and out-of-distribution datasets.
    Args:
        dataset_ind:
        dataset_ood:
        encoder:
        classifier:
        criterion:
        device:
        args:

    Returns:

    """
    train_idx, train_ood_idx = dataset_ind.train_mask, dataset_ood.node_idx
    logits_in, penultimate = encoder(dataset_ind.x, dataset_ind.edge_index)
    logits_in, penultimate = logits_in[train_idx], penultimate[train_idx]

    # sample_point, sample_edge, sample_label = generate_outliers(
    #     dataset_ind.x,
    #     device=device,
    #     num_nodes=len(logits_in),
    #     # num_nodes=dataset_ind.num_nodes,
    #     num_features=dataset_ind.num_features,
    #     num_edges=dataset_ind.num_edges,
    # )

    # ood = torch.cat([dataset_ood.x, sample_point])
    # sample_logits, sampling_penultimate = encoder(ood, dataset_ood.edge_index)
    sample_logits, sampling_penultimate = encoder(dataset_ood.x, dataset_ood.edge_index)
    sample_logits, sampling_penultimate = sample_logits[train_ood_idx], sampling_penultimate[train_ood_idx]

    input_for_lr_id = classifier(
        penultimate,
        dataset_ind.x,
        dataset_ind.edge_index,
        train_idx
    )

    input_for_lr_ood = classifier(
        sampling_penultimate,
        dataset_ood.x,
        dataset_ood.edge_index,
        train_ood_idx
    )

    min_length = min(len(input_for_lr_id), len(input_for_lr_ood))

    input_for_lr = torch.cat([input_for_lr_id[:min_length], input_for_lr_ood[:min_length]])

    labels_for_lr = torch.cat([
        torch.ones(min_length, device=device),
        torch.zeros(min_length, device=device)
    ])

    # print("训练ID")
    # ic(input_for_lr[:10])
    # ic(labels_for_lr[:10])
    # print("训练OOD")
    # ic(input_for_lr[-10:])
    # ic(labels_for_lr[-10:])

    sample_sup_loss = nn.BCELoss()(input_for_lr, labels_for_lr)

    pred_in = F.log_softmax(logits_in, dim=1)
    loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1))

    # ic(loss)
    # ic(0.5 * -(sample_logits.mean(1) - torch.logsumexp(sample_logits, dim=1)).mean())
    # ic(sample_sup_loss)

    # loss += 0.5 * -(sample_logits.mean(1) - torch.logsumexp(sample_logits, dim=1)).mean()
    if args.generate_ood:
        loss += sample_sup_loss

    return loss
