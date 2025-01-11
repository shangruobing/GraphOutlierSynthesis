import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

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
        self.classifier = Classifier(in_features=args.hidden_channels)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits, penultimate = self.encoder(x, edge_index)
        return logits

    def detect(self, dataset, node_idx, device, args):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits, penultimate = self.encoder(x, edge_index)
        return self.classifier(penultimate, dataset.x, dataset.edge_index, node_idx).squeeze()

    def loss_compute(self, dataset_ind: Data, dataset_ood: Data, synthesis_ood_dataset: Data, criterion, device, args):
        train_idx = dataset_ind.train_mask
        logits_in, penultimate = self.encoder(dataset_ind.x.to(device), dataset_ind.edge_index.to(device))
        logits_in, penultimate = logits_in[train_idx], penultimate[train_idx]
        pred_in = F.log_softmax(logits_in, dim=1)
        loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))
        return loss

    def classify_loss_compute(self, dataset_ind: Data, dataset_ood: Data, synthesis_ood_dataset: Data, criterion, device, args):
        return compute_loss(dataset_ind, dataset_ood, synthesis_ood_dataset, self.encoder, self.classifier, criterion, device, args)


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
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features * 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=in_features * 2, out_features=in_features // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=in_features // 2, out_features=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, logits, x, edge_index, mask):
        logits = self.classifier(logits)
        logits = logits.squeeze()
        return logits
