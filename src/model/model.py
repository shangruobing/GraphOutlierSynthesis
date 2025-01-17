import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul

from src.common.parse import Arguments
from src.model.backbone import GCN, MLP, GAT, MixHop, GCNJK, GATJK
from src.outlier.loss import compute_loss


class GNNOutlier(nn.Module):
    """
    The model class of energy-based models for out-of-distribution detection
    """

    def __init__(self, num_features, num_classes, args: Arguments):
        super(GNNOutlier, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(
                in_channels=num_features,
                hidden_channels=args.hidden_channels,
                out_channels=num_classes,
                num_layers=args.num_layers,
                dropout=args.dropout,
                use_bn=True
            )
        elif args.backbone == 'mlp':
            self.encoder = MLP(
                in_channels=num_features,
                hidden_channels=args.hidden_channels,
                out_channels=num_classes,
                num_layers=args.num_layers,
                dropout=args.dropout)

        elif args.backbone == 'gat':
            self.encoder = GAT(
                num_features,
                args.hidden_channels,
                num_classes,
                num_layers=args.num_layers,
                dropout=args.dropout,
                use_bn=True
            )
        elif args.backbone == 'mixhop':
            self.encoder = MixHop(
                num_features,
                args.hidden_channels,
                num_classes,
                num_layers=args.num_layers,
                dropout=args.dropout
            )
        elif args.backbone == 'gcnjk':
            self.encoder = GCNJK(
                num_features,
                args.hidden_channels,
                num_classes,
                num_layers=args.num_layers,
                dropout=args.dropout
            )
        elif args.backbone == 'gatjk':
            self.encoder = GATJK(
                num_features,
                args.hidden_channels,
                num_classes,
                num_layers=args.num_layers,
                dropout=args.dropout
            )
        else:
            raise NotImplementedError
        self.classifier = Classifier(
            in_features=num_features,
            hidden_channels=args.hidden_channels
        )

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits, penultimate = self.encoder(x, edge_index)
        return logits

    def detect(self, dataset, node_idx, device, args):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits, penultimate = self.encoder(x, edge_index)
        # penultimate = penultimate[node_idx]
        return self.classifier(penultimate, dataset.x, dataset.edge_index, node_idx).squeeze()

    @staticmethod
    def propagation(e, edge_index, prop_layers=1, alpha=0.5):
        """energy belief propagation, return the energy after propagation"""
        e = e.unsqueeze(1)
        N = e.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm = 1. / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        origin_device = e.device
        if str(e.device).startswith("mps"):
            e = e.cpu()
            adj = adj.cpu()
        for _ in range(prop_layers):
            e = e * alpha + matmul(adj, e) * (1 - alpha)
        e = e.to(origin_device)
        return e.squeeze(1)

    def loss_compute(self, dataset_ind: Data, dataset_ood: Data, synthesis_ood_dataset: Data, criterion, device, args):
        # train_idx = dataset_ind.train_mask
        # logits_in, penultimate = self.encoder(dataset_ind.x.to(device), dataset_ind.edge_index.to(device))
        # logits_in, penultimate = logits_in[train_idx], penultimate[train_idx]
        # pred_in = F.log_softmax(logits_in, dim=1)
        # loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))
        # return loss
        x_in, edge_index_in = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)
        x_out, edge_index_out = dataset_ood.x.to(device), dataset_ood.edge_index.to(device)

        # get predicted logits from gnn classifier
        logits_in, penultimate = self.encoder(x_in, edge_index_in)
        logits_out, penultimate = self.encoder(x_out, edge_index_out)

        train_in_idx, train_ood_idx = dataset_ind.train_mask, dataset_ood.node_idx

        # compute supervised training loss
        pred_in = F.log_softmax(logits_in[train_in_idx], dim=1)
        sup_loss = criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))

        energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1)
        energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1)

        energy_in = self.propagation(energy_in, edge_index_in, prop_layers=2, alpha=0.5)[train_in_idx]
        energy_out = self.propagation(energy_out, edge_index_out, prop_layers=2, alpha=0.5)[train_ood_idx]

        # truncate to have the same length
        if energy_in.shape[0] != energy_out.shape[0]:
            min_n = min(energy_in.shape[0], energy_out.shape[0])
            energy_in = energy_in[:min_n]
            energy_out = energy_out[:min_n]

        # compute regularization loss
        m_in = -5
        m_out = -1
        reg_loss = torch.mean(F.relu(energy_in - m_in) ** 2 + F.relu(m_out - energy_out) ** 2)

        lamda = 1
        loss = sup_loss + lamda * reg_loss

        return loss

    def classify_loss_compute(self, dataset_ind: Data, dataset_ood: Data, synthesis_ood_dataset: Data, criterion, device, args):
        return compute_loss(dataset_ind, dataset_ood, synthesis_ood_dataset, self.encoder, self.classifier, criterion, device, args)


class AttentionFusion(nn.Module):
    def __init__(self, x_dim, logits_dim, energy_dim, output_dim):
        """
        Initialize the attention fusion module.
        Args:
            x_dim: Dimensionality of node features x.
            logits_dim: Dimensionality of logits.
            energy_dim: Dimensionality of energy (typically 1).
            output_dim: Output dimensionality after fusion.
        """
        super().__init__()
        self.x_transform = nn.Linear(x_dim, output_dim)
        self.logits_transform = nn.Linear(logits_dim, output_dim)
        self.energy_transform = nn.Linear(energy_dim, output_dim)

        self.attention = nn.Sequential(
            nn.Linear(output_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, logits, energy):
        """
        Forward pass for attention-based feature fusion.
        Args:
            x: Node features, shape (N, x_dim).
            logits: Logits from the encoder, shape (N, logits_dim).
            energy: Energy values, shape (N, energy_dim).
        Returns:
            Fused features, shape (N, output_dim).
        """
        # Transform all inputs to the same dimension
        x_transformed = self.x_transform(x)  # (N, output_dim)
        logits_transformed = self.logits_transform(logits)  # (N, output_dim)
        energy_transformed = self.energy_transform(energy)  # (N, output_dim)

        # Compute attention weights for each component
        x_weight = self.attention(x_transformed)  # (N, 1)
        logits_weight = self.attention(logits_transformed)  # (N, 1)
        energy_weight = self.attention(energy_transformed)  # (N, 1)

        # Normalize attention weights
        total_weight = x_weight + logits_weight + energy_weight
        x_weight = x_weight / total_weight
        logits_weight = logits_weight / total_weight
        energy_weight = energy_weight / total_weight

        # Fuse the features
        fused = x_weight * x_transformed + logits_weight * logits_transformed + energy_weight * energy_transformed

        return fused


class Classifier(nn.Module):
    """
    A classifier for encoder penultimate output.
    """

    def __init__(self, in_features, hidden_channels, alpha=0.5, prop_layers=2):
        """
        Initialize a classifier for encoder output.
        Args:
            in_features: number of input features
        """
        super().__init__()
        self.alpha = alpha
        self.prop_layers = prop_layers
        self.feature_processor = nn.Sequential(
            nn.Linear(in_features, in_features * 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features * 2, in_features)
        )
        self.energy_processor = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8)
        )
        self.logit_processor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels // 2)
        )
        self.gcn = GCNConv(in_features, hidden_channels)
        self.gcn_transform = nn.Linear(hidden_channels, in_features)

        num_features = in_features // 2 + hidden_channels // 2 + 1

        self.attention_fusion = AttentionFusion(x_dim=in_features, logits_dim=hidden_channels // 2, energy_dim=8, output_dim=num_features)

        self.fusion = nn.Sequential(
            nn.Linear(num_features, hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def compute_energy(logits):
        """
        Compute energy using logsumexp over logits.
        Args:
            logits: model logits of shape (N, C)
        Returns:
            energy: computed energy of shape (N,)
        """
        return torch.logsumexp(logits, dim=-1)

    @staticmethod
    def propagate_energy(energy, edge_index, prop_layers, alpha):
        """
        Perform energy propagation over the graph.
        Args:
            energy: initial energy of shape (N,)
            edge_index: edge indices of the graph
            prop_layers: number of propagation layers
            alpha: propagation smoothing factor
        Returns:
            propagated energy of shape (N,)
        """
        """energy belief propagation, return the energy after propagation"""
        e = energy.unsqueeze(1)
        N = e.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm = 1. / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        origin_device = e.device
        if str(e.device).startswith("mps"):
            e = e.cpu()
            adj = adj.cpu()
        for _ in range(prop_layers):
            e = e * alpha + matmul(adj, e) * (1 - alpha)
        e = e.to(origin_device)
        return e.squeeze(1)

    def forward(self, logits, x, edge_index, mask):
        """
        Forward pass of the unified classifier.
        Args:
            logits: logits from the encoder, shape (N, C)
            x: input node features, shape (N, F)
            edge_index: edge indices of the graph
            mask: optional mask to select specific nodes
        Returns:
            predictions: final classification predictions
        """
        # Compute energy
        energy = self.compute_energy(logits)
        energy = self.propagate_energy(energy, edge_index, self.prop_layers, self.alpha)

        # Normalize energy
        energy = (energy - energy.mean()) / (energy.std() + 1e-6)

        # Process logits
        processed_x = self.feature_processor(x)  # Non-linear transformation of x
        processed_logits = self.logit_processor(logits)
        processed_energy = self.energy_processor(energy.unsqueeze(-1))

        graph_enhanced_x = self.gcn(processed_x, edge_index)
        graph_enhanced_x = self.gcn_transform(graph_enhanced_x)

        # Feature fusion with attention
        fused_features = self.attention_fusion(
            graph_enhanced_x, processed_logits, processed_energy
        )

        predictions = self.fusion(fused_features).squeeze()

        predictions = predictions[mask]

        return predictions
