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

    sample_point, sample_edge, sample_label = generate_outliers(
        dataset_ind.x,
        device=device,
        num_nodes=len(logits_in),
        # num_nodes=dataset_ind.num_nodes,
        num_features=dataset_ind.num_features,
        num_edges=dataset_ind.num_edges,
    )

    # ood_logits, ood_penultimate = encoder(dataset_ood.x, dataset_ood.edge_index)
    # sample_logits, sample_penultimate = encoder(sample_point, sample_edge)

    # input_for_lr = classifier(torch.cat([
    #     penultimate[:100],
    #     ood_penultimate[:100],
    #     sample_penultimate[:100],
    # ]))

    # ic(input_for_lr.shape)
    # ic(input_for_lr[:10])
    # ic(input_for_lr[110:120])
    # ic(input_for_lr[-10:])

    ood = torch.cat([dataset_ood.x, sample_point])
    sample_logits, sampling_penultimate = encoder(ood, dataset_ood.edge_index)
    sample_logits, sampling_penultimate = sample_logits[train_ood_idx], sampling_penultimate[train_ood_idx]

    min_length = min(len(logits_in), len(sample_logits))

    # torch.

    # ic(len(logits_in))
    # ic(len(sample_logits))

    input_for_lr = classifier(torch.cat([
        penultimate[:min_length], sampling_penultimate[torch.randperm(len(sample_logits))[:min_length]]
    ]))

    ic('training begin')
    # ic(input_for_lr.shape)
    # ic(penultimate[:min_length].shape)
    # ic(sampling_penultimate[:min_length].shape)
    # ic(input_for_lr[:10])
    # ic(input_for_lr[-10:])

    # input_for_lr = classifier(torch.cat(tensors=[penultimate, sampling_penultimate],
    #                                     dim=0))

    labels_for_lr = torch.cat([
        torch.ones(min_length, device=device),
        torch.zeros(min_length, device=device)
    ])

    # labels_for_lr = torch.cat([
    #     torch.ones(100, device=device),
    #     torch.zeros(200, device=device)
    # ])

    # ic(labels_for_lr.shape)
    print("训练ID")
    ic(input_for_lr[:10])
    ic(labels_for_lr[:10])
    print("训练OOD")
    ic(input_for_lr[-10:])
    ic(labels_for_lr[-10:])
    # ic('training end')

    # print("why")
    # print("aaaa",classifier(sampling_penultimate))

    criterion_BCE = nn.BCELoss()
    sample_sup_loss = criterion_BCE(input_for_lr, labels_for_lr)

    pred_in = F.log_softmax(logits_in, dim=1)
    loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1))

    # ic(loss)
    loss += 0.5 * -(sample_logits.mean(1) - torch.logsumexp(sample_logits, dim=1)).mean()
    if args.generate_ood:
        # ic(loss)
        # ic(sample_sup_loss)
        loss += sample_sup_loss

    # print("bbbb",classifier(sampling_penultimate))
    # print("loss id cla",id(classifier))

    return loss
