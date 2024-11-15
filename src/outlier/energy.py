import torch
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul

__all__ = ["energy_propagation"]


def energy_propagation(embeddings, edge_index, valid_index=None, num_prop_layers=1, alpha=0.5):
    """
    Energy belief propagation, return the energy after propagation
    Args:
        embeddings: The embeddings of the Encoder
        valid_index: Supervised Node index
        edge_index: Graph edge index
        num_prop_layers: Number of propagation layers
        alpha: The weight of the original embeddings

    Returns:

    """
    embeddings = embeddings.unsqueeze(1)
    N = embeddings.shape[0]
    row, col = edge_index

    # filter the out-of-bound nodes
    col = col[col < N]
    row = row[row < N]

    if valid_index is not None:
        valid = torch.nonzero(valid_index).squeeze()
        row_mask = torch.isin(row, valid)
        row = row[row_mask]
        col_mask = torch.isin(col, valid)
        col = col[col_mask]

    if row.shape[0] != col.shape[0]:
        min_length = min(row.shape[0], col.shape[0])
        row = row[:min_length]
        col = col[:min_length]

    d = degree(col, N).float()
    d_norm = 1. / d[col]
    value = torch.ones_like(row) * d_norm
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    for _ in range(num_prop_layers):
        embeddings = embeddings * alpha + matmul(adj, embeddings) * (1 - alpha)
    return embeddings.squeeze(1)
