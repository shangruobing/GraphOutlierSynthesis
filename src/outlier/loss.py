import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from icecream import ic

from src.outlier.energy import energy_propagation
from src.common.parse import Arguments

__all__ = ["compute_loss"]


def compute_loss(
        dataset_id: Data,
        dataset_ood: Data,
        synthesis_ood_dataset: Data,
        encoder: nn.Module,
        classifier: nn.Module,
        criterion: nn.NLLLoss,
        device: torch.device,
        args: Arguments
):
    """
    Compute the loss for in-distribution and out-of-distribution datasets.
    loss = supervised_learning_loss + energy_regularization_loss + classifier_loss
    Args:
        dataset_id: The in-distribution dataset
        dataset_ood: The out-of-distribution dataset
        synthesis_ood_dataset: The out-of-distribution dataset generated by KNN.
        encoder: The GNN encoder
        classifier: The penultimate embeddings classifier
        criterion: The loss function
        device: The device to run the computation
        args: The command line arguments

    Returns: The value of loss function.

    """
    loss = torch.tensor(data=0, dtype=torch.float, device=device)

    train_idx, train_ood_idx = dataset_id.train_mask, dataset_ood.node_idx

    # Get the GNN output for the ID data
    logits_id, penultimate_id = encoder(dataset_id.x, dataset_id.edge_index)
    logits_id, penultimate_id = logits_id[train_idx], penultimate_id[train_idx]

    # The loss is calculated using the ID output of the GNN
    predict_id = F.log_softmax(logits_id, dim=1)
    supervised_learning_loss = criterion(predict_id, dataset_id.y[train_idx].squeeze(1))
    loss += supervised_learning_loss

    # Whether to use ood
    if args.use_energy:
        # Get the GNN output for the OOD data
        logits_ood, penultimate_ood = encoder(dataset_ood.x, dataset_ood.edge_index)
        logits_ood, penultimate_ood = logits_ood[train_ood_idx], penultimate_ood[train_ood_idx]

        if args.synthesis_ood:
            train_idx = synthesis_ood_dataset.node_idx
            # Get the GNN output for the synthesised OOD data
            logits_knn_ood, penultimate_knn_ood = encoder(synthesis_ood_dataset.x, synthesis_ood_dataset.edge_index)
            logits_knn_ood, penultimate_knn_ood = logits_knn_ood[train_idx], penultimate_knn_ood[train_idx]
            logits_ood = torch.cat([logits_ood, logits_knn_ood])
            penultimate_ood = torch.cat([penultimate_ood, penultimate_knn_ood])

        # Calculate the energy scores of ID and OOD output by GNN
        # temperature for Softmax
        T = 1.0
        energy_id = - T * torch.logsumexp(logits_id / T, dim=-1)
        energy_ood = - T * torch.logsumexp(logits_ood / T, dim=-1)

        # Complete the propagation of energy
        if args.use_energy_propagation:
            # number of layers for energy belief propagation
            num_prop_layers = 1
            # weight for residual connection in propagation
            alpha = 0.5
            energy_id = energy_propagation(energy_id, dataset_id.edge_index, train_idx, num_prop_layers=num_prop_layers, alpha=alpha)
            energy_ood = energy_propagation(energy_ood, dataset_ood.edge_index, train_ood_idx, num_prop_layers=num_prop_layers, alpha=alpha)

        energy_id, energy_ood = trim_to_same_length(energy_id, energy_ood)

        # Calculate the energy regularization loss
        energy_regularization_loss = torch.mean(
            torch.pow(F.relu(energy_id - args.lower_bound_id), 2)
            +
            torch.pow(F.relu(args.upper_bound_id - energy_ood), 2)
        )
        loss += args.lamda * energy_regularization_loss

        if args.use_classifier:
            # The ID data is fed into the classifier
            classifier_id = classifier(
                penultimate_id,
                dataset_id.x,
                dataset_id.edge_index,
                train_idx
            )

            # The OOD data is fed into the classifier
            classifier_ood = classifier(
                penultimate_ood,
                dataset_ood.x,
                dataset_ood.edge_index,
                train_ood_idx
            )

            if args.use_energy_filter:
                # The classifier output is filtered using the energy score
                classifier_id, classifier_ood = filter_by_energy(
                    classifier_id=classifier_id,
                    classifier_ood=classifier_ood,
                    energy_id=energy_id,
                    energy_ood=energy_ood,
                    upper_bound_id=args.upper_bound_id,
                    lower_bound_id=args.lower_bound_id
                )

            # Construct the classifier outputs and labels
            min_length = min(len(classifier_id), len(classifier_ood))
            if min_length == 0:
                raise ValueError("No data left after filtering by energy. Adjust the upper_bound_id or lower_bound_ood.")
            classifier_output = torch.cat([classifier_id[:min_length], classifier_ood[:min_length]])
            classifier_label = torch.cat([
                torch.ones(min_length, device=device),
                torch.zeros(min_length, device=device)
            ])

            classifier_loss = nn.BCELoss()(classifier_output, classifier_label)
            loss += args.delta * classifier_loss

    return loss


def filter_by_energy(
        classifier_id: torch.Tensor,
        classifier_ood: torch.Tensor,
        energy_id: torch.Tensor,
        energy_ood: torch.Tensor,
        upper_bound_id=-5,
        lower_bound_id=-1
):
    """
    Filter the classifier output by energy scores.
    Args:
        classifier_id:
        classifier_ood:
        energy_id:
        energy_ood:
        upper_bound_id:
        lower_bound_id:

    Returns:
    """
    filtered_classifier_ood_index = torch.nonzero(energy_ood > upper_bound_id).squeeze().view(-1)
    filtered_classifier_id_index = torch.nonzero(energy_id > lower_bound_id).squeeze().view(-1)

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
