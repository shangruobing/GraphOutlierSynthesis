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

    min_length = min(len(logits_in), len(sample_logits))

    # print(penultimate[0])
    # print(sampling_penultimate[0])
    input_for_lr = classifier(torch.cat([
        penultimate[:min_length], sampling_penultimate[:min_length]
        # penultimate[:min_length], sampling_penultimate[torch.randperm(len(sample_logits))[:min_length]]
    ]))

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

    # ic(torch.argmax(input_for_lr[-10:], dim=1))

    # criterion_BCE = nn.CrossEntropyLoss()
    # sample_sup_loss = criterion_BCE(torch.argmax(input_for_lr, dim=1).float(), labels_for_lr)
    sample_sup_loss = nn.BCELoss()(input_for_lr, labels_for_lr)

    pred_in = F.log_softmax(logits_in, dim=1)
    loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1))

    loss += 0.5 * -(sample_logits.mean(1) - torch.logsumexp(sample_logits, dim=1)).mean()
    if args.generate_ood:
        loss += sample_sup_loss

    return loss
