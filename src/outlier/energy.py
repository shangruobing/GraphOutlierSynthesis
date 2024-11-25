import torch
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul

__all__ = ["energy_propagation"]


def energy_propagation(embeddings, edge_index, num_prop_layers=1, alpha=0.5):
    """
    Energy belief propagation, return the energy after propagation
    Args:
        embeddings: The embeddings of the Encoder
        edge_index: Graph edge index
        num_prop_layers: Number of propagation layers
        alpha: The weight of the original embeddings

    Returns:

    """
    embeddings = embeddings.unsqueeze(1)
    N = embeddings.shape[0]
    row, col = edge_index

    d = degree(col, N).float()
    d_norm = 1. / d[col]
    value = torch.ones_like(row) * d_norm
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    for _ in range(num_prop_layers):
        embeddings = embeddings * alpha + matmul(adj, embeddings) * (1 - alpha)
    return embeddings.squeeze(1)
