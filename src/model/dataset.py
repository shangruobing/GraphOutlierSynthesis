from typing import Tuple

import numpy as np
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Twitch
from torch_geometric.utils import stochastic_blockmodel_graph

from src.common.parse import Arguments
from src.model.data_utils import rand_splits
from src.outlier.knn import generate_outliers

__all__ = ["load_dataset", "create_knn_dataset"]


def load_dataset(args: Arguments) -> Tuple[Data, Data, Data]:
    """
    Load dataset according to the dataset name and ood type
    Args:
        args: arguments

    Returns:
        dataset_ind: in-distribution training dataset
        dataset_ood_tr: ood-distribution training dataset as ood exposure
        dataset_ood_te: a list of ood testing datasets or one ood testing dataset

    """
    # multi-graph datasets, use one as ind, the other as ood
    if args.dataset == 'twitch':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_twitch_dataset(args.data_dir)

    # single graph, use partial nodes as ind, others as ood according to domain info
    elif args.dataset == 'arxiv':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_arxiv_dataset(args.data_dir)

    # single graph, use original as ind, modified graphs as ood
    elif args.dataset in [
        'cora',
        'amazon-photo',
        'coauthor-cs',
    ]:
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_graph_dataset(args.data_dir, args.dataset, args.ood_type)

    else:
        raise NotImplementedError(f"Unsupported dataset {args.dataset}")

    if len(dataset_ind.y.shape) == 1:
        dataset_ind.y = dataset_ind.y.unsqueeze(1)
    if len(dataset_ood_tr.y.shape) == 1:
        dataset_ood_tr.y = dataset_ood_tr.y.unsqueeze(1)
    if len(dataset_ood_te.y.shape) == 1:
        dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)

    return dataset_ind, dataset_ood_tr, dataset_ood_te


def add_mask_property(dataset: Data) -> Data:
    """
    For each dataset, we add the following properties:
        node_idx: The index of nodes in the dataset
        train_mask: The mask for training nodes
        val_mask: The mask for validation nodes
        test_mask: The mask for testing nodes
    Args:
        dataset:

    Returns:

    """
    if hasattr(dataset, "node_idx"):
        print("Use the node_idx provided with the dataset.")
    else:
        print("The node_idx is not provided for the dataset. Use dataset.num_nodes.")
        dataset.node_idx = torch.arange(dataset.num_nodes)

    if hasattr(dataset, "train_mask") and hasattr(dataset, "val_mask") and hasattr(dataset, "test_mask"):
        print("Use the train_mask, val_mask and test_mask provided with the dataset.")
    else:
        print("The train_mask is not provided for the dataset. Use random splits.")
        train_mask, val_mask, test_mask = rand_splits(dataset.num_nodes)
        dataset.train_mask = train_mask
        dataset.val_mask = val_mask
        dataset.test_mask = test_mask
    return dataset


def load_twitch_dataset(data_dir) -> Tuple[Data, Data, Data]:
    transform = T.NormalizeFeatures()
    subgraph_names = ['DE', 'EN', 'ES', 'FR', 'RU']
    train_idx, valid_idx, test_idx = 0, 1, 2

    torch_dataset = Twitch(root=f'{data_dir}/Twitch', name=subgraph_names[train_idx], transform=transform)
    dataset_ind = add_mask_property(torch_dataset[0])

    torch_dataset = Twitch(root=f'{data_dir}/Twitch', name=subgraph_names[valid_idx], transform=transform)
    dataset_ood_tr = add_mask_property(torch_dataset[0])

    torch_dataset = Twitch(root=f'{data_dir}/Twitch', name=subgraph_names[test_idx], transform=transform)
    dataset_ood_te = add_mask_property(torch_dataset[0])

    return dataset_ind, dataset_ood_tr, dataset_ood_te


def load_arxiv_dataset(data_dir) -> Tuple[Data, Data, Data]:
    """
    This dataset contains Arxiv from 2013 to 2020.
    We use the data before 2015 as in-distribution data, the data from 2016 to 2017 as OOD training data, and the data from 2018 to 2020 as OOD testing data.
    :param data_dir:
    :return:
    """
    ogb_dataset = NodePropPredDataset(root=f'{data_dir}/ogb', name='ogbn-arxiv')
    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    node_feat = torch.as_tensor(ogb_dataset.graph['node_feat'])
    label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    year = ogb_dataset.graph['node_year']

    center_node_mask = (year <= 2015).squeeze(1)
    dataset_ind = Data(x=node_feat, edge_index=edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[center_node_mask]
    dataset_ind = add_mask_property(dataset_ind)

    center_node_mask = (year > 2015).squeeze(1) * (year <= 2017).squeeze(1)
    dataset_ood_tr = Data(x=node_feat, edge_index=edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ood_tr.node_idx = idx[center_node_mask]
    dataset_ood_tr = add_mask_property(dataset_ood_tr)

    center_node_mask = (year > 2017).squeeze(1) * (year <= 2020).squeeze(1)
    dataset_ood_te = Data(x=node_feat, edge_index=edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ood_te.node_idx = idx[center_node_mask]
    dataset_ood_te = add_mask_property(dataset_ood_te)

    return dataset_ind, dataset_ood_tr, dataset_ood_te


def load_graph_dataset(data_dir, dataset_name, ood_type) -> Tuple[Data, Data, Data]:
    """
    single graph, use original as ind, modified graphs as ood
    Args:
        data_dir:
        dataset_name:
        ood_type:

    Returns:

    """
    transform = T.NormalizeFeatures()
    if dataset_name == 'cora':
        torch_dataset = Planetoid(root=f'{data_dir}/Planetoid', split='public', name=dataset_name, transform=transform)
    elif dataset_name == 'amazon-photo':
        torch_dataset = Amazon(root=f'{data_dir}/Amazon', name='Photo', transform=transform)
    elif dataset_name == 'coauthor-cs':
        torch_dataset = Coauthor(root=f'{data_dir}/Coauthor', name='CS', transform=transform)
    else:
        raise NotImplementedError

    dataset = torch_dataset[0]
    dataset = add_mask_property(dataset)

    if ood_type == 'structure':
        dataset_ood_tr = create_structure_manipulation_dataset(dataset)
        dataset_ood_te = create_structure_manipulation_dataset(dataset)
    elif ood_type == 'feature':
        dataset_ood_tr = create_feature_interpolation_dataset(dataset)
        dataset_ood_te = create_feature_interpolation_dataset(dataset)
    elif ood_type == 'label':
        dataset, dataset_ood_tr, dataset_ood_te = create_label_leave_out_dataset(dataset)
    else:
        raise NotImplementedError

    return dataset, dataset_ood_tr, dataset_ood_te


def create_structure_manipulation_dataset(data, p_ii=1.5, p_ij=0.5) -> Data:
    """
    Structure manipulation: use the original graph as in-distribution data and adopt stochastic block model to randomly generate a graph for OOD data.
    :param data:
    :param p_ii:
    :param p_ij:
    :return:
    """
    n = data.num_nodes
    d = data.edge_index.size(1) / data.num_nodes / (data.num_nodes - 1)
    num_blocks = int(data.y.max()) + 1
    p_ii, p_ij = p_ii * d, p_ij * d
    block_size = n // num_blocks
    block_sizes = [block_size for _ in range(num_blocks - 1)] + [block_size + n % block_size]
    edge_probs = torch.ones((num_blocks, num_blocks)) * p_ij
    edge_probs[torch.arange(num_blocks), torch.arange(num_blocks)] = p_ii
    edge_index = stochastic_blockmodel_graph(block_sizes, edge_probs)
    dataset = Data(x=data.x, edge_index=edge_index, y=data.y)
    dataset.node_idx = torch.arange(n)
    return dataset


def create_feature_interpolation_dataset(data) -> Data:
    """
    Feature interpolation: use random interpolation to create node features for OOD data and the original graph as in-distribution data.
    :param data:
    :return:
    """
    x = data.x
    n = data.num_nodes
    idx = torch.randint(0, n, (n, 2))
    weight = torch.rand(n).unsqueeze(1)
    x_new = x[idx[:, 0]] * weight + x[idx[:, 1]] * (1 - weight)
    dataset = Data(x=x_new, edge_index=data.edge_index, y=data.y)
    dataset.node_idx = torch.arange(n)
    return dataset


def create_label_leave_out_dataset(dataset) -> Data:
    """
    Label leave-out: use nodes with partial classes as in-distribution and leave out others for OOD.
    :param dataset:
    :return:
    """
    label = dataset.y
    unique_elements = torch.unique(label)
    class_t = int(np.median(unique_elements))

    center_node_mask_ind = (label > class_t)
    idx = torch.arange(label.size(0))
    dataset.node_idx = idx[center_node_mask_ind]

    for key in ["train_mask", "val_mask", "test_mask"]:
        mask = torch.zeros(label.size(0), dtype=torch.bool)
        mask[torch.as_tensor(dataset[key])] = True
        dataset[key] = idx[mask * center_node_mask_ind]

    dataset_ood_tr = Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y)
    dataset_ood_te = Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y)

    center_node_mask_ood_tr = (label == class_t)
    center_node_mask_ood_te = (label < class_t)
    dataset_ood_tr.node_idx = idx[center_node_mask_ood_tr]
    dataset_ood_te.node_idx = idx[center_node_mask_ood_te]
    return dataset, dataset_ood_tr, dataset_ood_te


def create_knn_dataset(
        data: Data,
        cov_mat=0.1,
        sampling_ratio=1.0,
        boundary_ratio=0.1,
        boundary_sampling_ratio=0.5,
        k=100,
        device=torch.device("cpu")
) -> Data:
    """

    Args:
        data: dataset
        cov_mat: The weight before the covariance matrix to determine the sampling range
        sampling_ratio: How many OOD samples to generate
        boundary_ratio: How many ID samples are defined as points near the boundary
        boundary_sampling_ratio: How many boundary used to generate outliers
        k: The number of nearest neighbors to return
        device: torch device

    Returns:

    """
    outlier = generate_outliers(
        data.x,
        num_nodes=data.num_nodes,
        num_features=data.num_features,
        num_edges=data.num_edges,
        cov_mat=cov_mat,
        sampling_ratio=sampling_ratio,
        boundary_ratio=boundary_ratio,
        boundary_sampling_ratio=boundary_sampling_ratio,
        k=k,
        device=device
    )
    dataset = Data(x=outlier.sample_points, edge_index=outlier.sample_edges, y=outlier.sample_labels)
    dataset.node_idx = torch.arange(len(outlier.sample_points))
    dataset = add_mask_property(dataset)
    return dataset
