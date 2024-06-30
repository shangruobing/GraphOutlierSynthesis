from typing import Tuple

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Twitch, WikiCS, Actor, WebKB, GitHub
from torch_geometric.utils import stochastic_blockmodel_graph

from src.common.parse import Arguments
from src.outlier.knn import generate_outliers


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

    # single graph, use original as ind, modified graphs as ood
    elif args.dataset in [
        'cora',
        "actor",
        'citeseer',
        'pubmed',
        'amazon-photo',
        'amazon-computer',
        'coauthor-cs',
        'coauthor-physics',
        "wiki-cs",
        "webkb",
        "github"
    ]:
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_graph_dataset(args.data_dir, args.dataset, args.ood_type)

    else:
        raise NotImplementedError

    # visualize(torch.tensor(dataset_ind.x), color=torch.tensor(dataset_ind.y), epoch=1)
    # visualize(torch.tensor(dataset_ood_tr.x), color=torch.tensor(dataset_ood_tr.y), epoch=1)
    # visualize(torch.tensor(dataset_ood_te.x), color=torch.tensor(dataset_ood_te.y), epoch=1)

    # print(len(dataset_ind.x))
    # print(len(dataset_ood_te.x))

    # visualize(
    #     torch.cat([
    #         torch.tensor(dataset_ind.x),
    #         torch.tensor(dataset_ood_te.x),
    #     ]),
    #     color=
    #     torch.cat([
    #         torch.ones(len(dataset_ind.x)),
    #         torch.zeros(len(dataset_ood_te.x))
    #     ])
    #     , epoch=1)

    # visualize(
    #     torch.cat([
    #         torch.tensor(dataset_ind.x),
    #         torch.tensor(dataset_ood_te.x),
    #     ]),
    #     color=
    #     torch.cat([
    #         torch.ones(len(dataset_ind.x)),
    #         torch.zeros(len(dataset_ood_te.x))
    #     ])
    #     , epoch=1)

    return dataset_ind, dataset_ood_tr, dataset_ood_te


def load_twitch_dataset(data_dir):
    transform = T.NormalizeFeatures()
    subgraph_names = ['DE', 'EN', 'ES', 'FR', 'RU']
    train_idx, valid_idx = 0, 1
    dataset_ood_te = []
    for i in range(len(subgraph_names)):
        torch_dataset = Twitch(root=f'{data_dir}/Twitch',
                               name=subgraph_names[i], transform=transform)
        dataset = torch_dataset[0]
        dataset.node_idx = torch.arange(dataset.num_nodes)
        if i == train_idx:
            dataset_ind = dataset
        elif i == valid_idx:
            dataset_ood_tr = dataset
        else:
            dataset_ood_te.append(dataset)

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
    if dataset_name in ('cora', 'citeseer', 'pubmed'):
        torch_dataset = Planetoid(root=f'{data_dir}/Planetoid', split='public', name=dataset_name, transform=transform)
    elif dataset_name == 'amazon-photo':
        torch_dataset = Amazon(root=f'{data_dir}/Amazon', name='Photo', transform=transform)
    elif dataset_name == 'amazon-computer':
        torch_dataset = Amazon(root=f'{data_dir}/Amazon', name='Computers', transform=transform)
    elif dataset_name == 'coauthor-cs':
        torch_dataset = Coauthor(root=f'{data_dir}/Coauthor', name='CS', transform=transform)
    elif dataset_name == 'coauthor-physics':
        torch_dataset = Coauthor(root=f'{data_dir}/Coauthor', name='Physics', transform=transform)
    elif dataset_name == 'wiki-cs':
        torch_dataset = WikiCS(root=f'{data_dir}/WikiCS', transform=transform, is_undirected=False)
    elif dataset_name == 'actor':
        torch_dataset = Actor(root=f'{data_dir}/Actor', transform=transform)
    elif dataset_name == 'webkb':
        torch_dataset = WebKB(name="Cornell", root=f'{data_dir}/WebKB', transform=transform)
    elif dataset_name == 'github':
        torch_dataset = GitHub(root=f'{data_dir}/GitHub', transform=transform)
    else:
        raise NotImplementedError

    # print("Have train_mask", len(dataset.train_mask) > 0)

    dataset = torch_dataset[0]
    dataset.node_idx = torch.arange(dataset.num_nodes)
    dataset_ind = dataset

    if ood_type == 'structure':
        dataset_ood_tr = create_sbm_dataset(dataset)
        dataset_ood_te = create_sbm_dataset(dataset)
    elif ood_type == 'feature':
        dataset_ood_tr = create_feat_noise_dataset(dataset)
        dataset_ood_te = create_feat_noise_dataset(dataset)
    elif ood_type == 'label':
        dataset_ood_tr = create_label_leave_out_dataset(dataset, dataset_ind, dataset_name)
        dataset_ood_te = create_label_leave_out_dataset(dataset, dataset_ind, dataset_name)
    elif ood_type == 'knn':
        dataset_ood_tr = create_knn_dataset(dataset)
        dataset_ood_te = create_knn_dataset(dataset)
    else:
        raise NotImplementedError
    return dataset_ind, dataset_ood_tr, dataset_ood_te


def create_sbm_dataset(data, p_ii=1.5, p_ij=0.5) -> Data:
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


def create_feat_noise_dataset(data) -> Data:
    x = data.x
    n = data.num_nodes
    idx = torch.randint(0, n, (n, 2))
    weight = torch.rand(n).unsqueeze(1)
    x_new = x[idx[:, 0]] * weight + x[idx[:, 1]] * (1 - weight)

    dataset = Data(x=x_new, edge_index=data.edge_index, y=data.y)
    dataset.node_idx = torch.arange(n)
    return dataset


def create_label_leave_out_dataset(dataset, dataset_ind, dataset_name) -> Data:
    label = dataset.y
    unique_elements = torch.unique(label)
    class_t = int(np.median(unique_elements))

    center_node_mask_ind = (label > class_t)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[center_node_mask_ind]

    if dataset_name in ('cora', 'citeseer', 'pubmed', 'arxiv', "actor"):
        split_idx = ["train_mask", "val_mask", "test_mask"]
        tensor_split_idx = {}
        idx = torch.arange(label.size(0))
        for key in split_idx:
            mask = torch.zeros(label.size(0), dtype=torch.bool)
            mask[torch.as_tensor(split_idx[key])] = True
            tensor_split_idx[key] = idx[mask * center_node_mask_ind]
        dataset_ind.splits = tensor_split_idx

    dataset_ood_te = Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y)
    center_node_mask_ood_te = (label <= class_t)
    dataset_ood_te.node_idx = idx[center_node_mask_ood_te]
    return dataset_ood_te


def create_knn_dataset(data) -> Data:
    sample_point, sample_edge, sample_label = generate_outliers(
        data.x,
        num_nodes=data.num_nodes,
        num_features=data.num_features,
        num_edges=data.num_edges,
    )
    dataset = Data(x=sample_point, edge_index=sample_edge, y=sample_label)
    dataset.node_idx = torch.arange(len(sample_point))
    return dataset
