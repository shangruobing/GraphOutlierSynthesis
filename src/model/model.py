# import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn.conv import GCNConv

from src.common.parse import Arguments
from src.model.backbone import GCN, MLP, GAT, MixHop, GCNJK, GATJK
# from src.outlier.energy import energy_propagation
from src.outlier.loss import compute_loss


# from torch_geometric.utils import degree
# from torch_sparse import SparseTensor, matmul


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
        self.classifier = Classifier(in_features=args.hidden_channels)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        """return predicted logits"""
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits, penultimate = self.encoder(x, edge_index)
        return logits

    def detect(self, dataset, node_idx, device, args):
        """return negative energy, a vector for all input nodes"""
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits, penultimate = self.encoder(x, edge_index)
        # logits, penultimate = logits[node_idx], penultimate[node_idx]
        # neg_energy = 1.0 * torch.logsumexp(logits / 1.0, dim=-1)
        # parser.add_argument('--K', type=int, default=2, help='number of layers for energy belief propagation')
        # K = 2
        # parser.add_argument('--alpha', type=float, default=0.5, help='weight for residual connection in propagation')
        # alpha = 0.5
        # if args.use_energy_propagation:
        #     neg_energy = energy_propagation(neg_energy, edge_index, num_prop_layers=K, alpha=alpha)
        # return neg_energy[node_idx]
        # logits, penultimate = self.encoder(dataset.x, dataset.edge_index)
        return self.classifier(penultimate, dataset.x, dataset.edge_index, node_idx).squeeze()

    def loss_compute(self, dataset_ind: Data, dataset_ood: Data, synthesis_ood_dataset: Data, criterion, device, args):
        return compute_loss(dataset_ind, dataset_ood, synthesis_ood_dataset, self.encoder, self.classifier, criterion, device, args)


class GCNEncoder(nn.Module):
    """
    An encoder for feature extraction.
    """

    def __init__(self, in_channels):
        """
        Initialize a. encoder for feature extraction.
        Args:
            in_channels: number of input channels
        """
        super().__init__()
        self.convs = nn.ModuleList([
            GCNConv(in_channels=in_channels, out_channels=in_channels // 2),
            GCNConv(in_channels=in_channels // 2, out_channels=in_channels // 4),
            GCNConv(in_channels=in_channels // 4, out_channels=2),
        ])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
        return x


class Classifier(nn.Module):
    """
    A classifier for encoder penultimate output.
    """

    def __init__(self, in_features):
        """
        Initialize a classifier for encoder output.
        Args:
            in_features: number of input features
        """
        super().__init__()
        self.encoder = GCNEncoder(in_channels=in_features)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128, out_features=in_features, bias=True),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=in_features // 2, bias=True),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features // 2, out_features=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, logits, x, edge_index, mask):
        # logits = self.encoder(logits, edge_index)
        logits = self.classifier(logits)
        logits = logits.squeeze()
        return logits
