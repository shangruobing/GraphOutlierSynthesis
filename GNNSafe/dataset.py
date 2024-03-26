from argparse import Namespace
from typing import Tuple

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Twitch, WikiCS, Actor, WebKB, GitHub
from torch_geometric.utils import stochastic_blockmodel_graph, subgraph

from OutliersGenerate.KNN import generate_outliers
from OutliersGenerate.test import visualize
from data_utils import to_sparse_tensor


def load_dataset(args: Namespace) -> Tuple[Data, Data]:
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
    elif args.dataset in 'arxiv':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_arxiv_dataset(args.data_dir)
    elif args.dataset in 'proteins':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_proteins_dataset(args.data_dir)

    # single graph, use original as ind, modified graphs as ood
    elif args.dataset in [
        'cora', "actor",
        'citeseer', 'pubmed',
        'amazon-photo', 'amazon-computer', 'coauthor-cs',
        'coauthor-physics', "wiki-cs", "actor", "webkb",
        "github"
    ]:
        dataset_ind, dataset_ood_te = load_graph_dataset(args.data_dir, args.dataset, args.ood_type)

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

    return dataset_ind, dataset_ood_te


def load_twitch_dataset(data_dir):
    transform = T.NormalizeFeatures()
    subgraph_names = ['DE', 'EN', 'ES', 'FR', 'RU']
    train_idx, valid_idx = 0, 1
    dataset_ood_te = []
    for i in range(len(subgraph_names)):
        torch_dataset = Twitch(root=f'{data_dir}Twitch',
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


def load_arxiv_dataset(data_dir, time_bound=[2015, 2017], inductive=True):
    from ogb.nodeproppred import NodePropPredDataset

    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv', root=f'{data_dir}/ogb')
    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    node_feat = torch.as_tensor(ogb_dataset.graph['node_feat'])
    label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    year = ogb_dataset.graph['node_year']

    year_min, year_max = time_bound[0], time_bound[1]
    test_year_bound = [2017, 2018, 2019, 2020]

    center_node_mask = (year <= year_min).squeeze(1)
    if inductive:
        ind_edge_index, _ = subgraph(center_node_mask, edge_index)
    else:
        ind_edge_index = edge_index

    dataset_ind = Data(x=node_feat, edge_index=ind_edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[center_node_mask]

    center_node_mask = (year <= year_max).squeeze(1) * (year > year_min).squeeze(1)
    if inductive:
        all_node_mask = (year <= year_max).squeeze(1)
        ood_tr_edge_index, _ = subgraph(all_node_mask, edge_index)
    else:
        ood_tr_edge_index = edge_index

    dataset_ood_tr = Data(x=node_feat, edge_index=ood_tr_edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ood_tr.node_idx = idx[center_node_mask]

    dataset_ood_te = []
    for i in range(len(test_year_bound) - 1):
        center_node_mask = (year <= test_year_bound[i + 1]).squeeze(1) * (year > test_year_bound[i]).squeeze(1)
        if inductive:
            all_node_mask = (year <= test_year_bound[i + 1]).squeeze(1)
            ood_te_edge_index, _ = subgraph(all_node_mask, edge_index)
        else:
            ood_te_edge_index = edge_index

        dataset = Data(x=node_feat, edge_index=ood_te_edge_index, y=label)
        idx = torch.arange(label.size(0))
        dataset.node_idx = idx[center_node_mask]
        dataset_ood_te.append(dataset)

    return dataset_ind, dataset_ood_tr, dataset_ood_te


def load_proteins_dataset(data_dir, inductive=True):
    from ogb.nodeproppred import NodePropPredDataset

    ogb_dataset = NodePropPredDataset(name='ogbn-proteins', root=f'{data_dir}/ogb')

    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    edge_feat = torch.as_tensor(ogb_dataset.graph['edge_feat'])
    label = torch.as_tensor(ogb_dataset.labels)

    edge_index_ = to_sparse_tensor(edge_index, edge_feat, ogb_dataset.graph['num_nodes'])
    node_feat = edge_index_.mean(dim=1)

    node_species = torch.as_tensor(ogb_dataset.graph['node_species'])
    species = [0] + node_species.unique().tolist()
    ind_species_min, ind_species_max = species[0], species[3]
    ood_tr_species_min, ood_tr_species_max = species[3], species[5]
    ood_te_species = [species[i] for i in range(5, 8)]

    center_node_mask = (node_species <= ind_species_max).squeeze(1) * (node_species > ind_species_min).squeeze(1)
    if inductive:
        all_node_mask = (node_species <= ind_species_max).squeeze(1)
        ind_edge_index, _ = subgraph(all_node_mask, edge_index)
    else:
        ind_edge_index = edge_index

    dataset_ind = Data(x=node_feat, edge_index=ind_edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[center_node_mask]

    center_node_mask = (node_species <= ood_tr_species_max).squeeze(1) * (node_species > ood_tr_species_min).squeeze(1)
    if inductive:
        all_node_mask = (node_species <= ood_tr_species_max).squeeze(1)
        ood_tr_edge_index, _ = subgraph(all_node_mask, edge_index)
    else:
        ood_tr_edge_index = edge_index

    dataset_ood_tr = Data(x=node_feat, edge_index=ood_tr_edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ood_tr.node_idx = idx[center_node_mask]

    dataset_ood_te = []
    for i in ood_te_species:
        center_node_mask = (node_species == i).squeeze(1)
        dataset = Data(x=node_feat, edge_index=edge_index, y=label)
        idx = torch.arange(label.size(0))
        dataset.node_idx = idx[center_node_mask]
        dataset_ood_te.append(dataset)

    return dataset_ind, dataset_ood_tr, dataset_ood_te


def load_graph_dataset(data_dir, dataset_name, ood_type):
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
        torch_dataset = Planetoid(root=f'{data_dir}Planetoid', split='public', name=dataset_name, transform=transform)
    elif dataset_name == 'amazon-photo':
        torch_dataset = Amazon(root=f'{data_dir}Amazon', name='Photo', transform=transform)
    elif dataset_name == 'amazon-computer':
        torch_dataset = Amazon(root=f'{data_dir}Amazon', name='Computers', transform=transform)
    elif dataset_name == 'coauthor-cs':
        torch_dataset = Coauthor(root=f'{data_dir}Coauthor', name='CS', transform=transform)
    elif dataset_name == 'coauthor-physics':
        torch_dataset = Coauthor(root=f'{data_dir}Coauthor', name='Physics', transform=transform)
    elif dataset_name == 'wiki-cs':
        torch_dataset = WikiCS(root=f'{data_dir}WikiCS', transform=transform, is_undirected=False)
    elif dataset_name == 'actor':
        torch_dataset = Actor(root=f'{data_dir}Actor', transform=transform)
    elif dataset_name == 'webkb':
        torch_dataset = WebKB(name="Cornell", root=f'{data_dir}WebKB', transform=transform)
    elif dataset_name == 'github':
        torch_dataset = GitHub(root=f'{data_dir}GitHub', transform=transform)
    else:
        raise NotImplementedError

    # print("Have train_mask", len(dataset.train_mask) > 0)

    dataset = torch_dataset[0]
    dataset.node_idx = torch.arange(dataset.num_nodes)
    dataset_ind = dataset

    if ood_type == 'structure':
        dataset_ood_te = create_sbm_dataset(dataset, p_ii=1.5, p_ij=0.5)
    elif ood_type == 'feature':
        dataset_ood_te = create_feat_noise_dataset(dataset)
    elif ood_type == 'label':
        dataset_ood_te = create_label_leave_out_dataset(dataset, dataset_ind, dataset_name)
    else:
        raise NotImplementedError
    return dataset_ind, dataset_ood_te


def create_sbm_dataset(data, p_ii=1.5, p_ij=0.5):
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


def create_feat_noise_dataset(data):
    x = data.x
    n = data.num_nodes
    idx = torch.randint(0, n, (n, 2))
    weight = torch.rand(n).unsqueeze(1)
    x_new = x[idx[:, 0]] * weight + x[idx[:, 1]] * (1 - weight)

    dataset = Data(x=x_new, edge_index=data.edge_index, y=data.y)
    dataset.node_idx = torch.arange(n)
    return dataset


def create_label_leave_out_dataset(dataset, dataset_ind, dataset_name):
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
