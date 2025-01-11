import argparse
from dataclasses import dataclass

from src.common.config import DATASET_PATH

__all__ = ["init_parser_args", "Arguments"]


def parser_add_main_args(parser):
    # setup and protocol
    parser.add_argument('--dataset', type=str, default='cora', choices=['twitch', 'arxiv', 'cora', 'amazon-photo', 'coauthor-cs'])
    parser.add_argument('--ood_type', type=str, default='structure', choices=['structure', 'feature', 'label'])
    parser.add_argument('--data_dir', type=str, default=str(DATASET_PATH))
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)

    # model network
    parser.add_argument('--method', type=str, default='gnnsafe', choices=[
        'MSP', 'OE', 'ODIN', 'Mahalanobis', 'MaxLogits', 'EnergyModel', 'EnergyProp', 'GNNSafe', 'GNNOutlier'
    ])
    parser.add_argument('--backbone', type=str, default='gcn', choices=[
        'gcn', 'mlp', 'gat', 'mixhop', 'gcnjk', 'gatjk', 'appnp', 'sgc'
    ])
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers for GNN classifiers')

    # training
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)

    # generate outliers
    parser.add_argument('--use_energy', action='store_true')
    parser.add_argument('--use_energy_propagation', action='store_true')
    parser.add_argument('--use_classifier', action='store_true')
    parser.add_argument('--use_energy_filter', action='store_true')
    parser.add_argument('--synthesis_ood', action='store_true')

    # hyper parameter
    parser.add_argument('--upper_bound_id', type=float, default=-5, help='upper bound for in-distribution energy')
    parser.add_argument('--lower_bound_id', type=float, default=-1, help='lower bound for in-distribution energy')
    parser.add_argument('--lamda', type=float, default=1.0, help='weight for regularization')
    parser.add_argument('--delta', type=float, default=1.0, help='weight for classifier loss')

    # samping strategy
    parser.add_argument('--cov_mat', type=float, default=0.1, help='The weight before the covariance matrix to determine the sampling range')
    parser.add_argument('--sampling_ratio', type=float, default=1.0, help='Number of OOD samples to generate')
    parser.add_argument('--boundary_ratio', type=float, default=0.1, help='Number of ID samples to pick to define as points near the boundary')
    parser.add_argument('--boundary_sampling_ratio', type=float, default=0.5, help='Number of boundary used to generate outliers')
    parser.add_argument('--k', type=int, default=100, help='Number of nearest neighbors to return')

    parser.add_argument('--T', type=float, default=1.0, help='temperature for Softmax')


@dataclass
class Arguments:
    string: str
    dataset: str
    ood_type: str
    data_dir: str
    device: int
    cpu: bool
    seed: int
    epochs: int
    method: str
    backbone: str
    hidden_channels: int
    num_layers: int
    weight_decay: float
    dropout: float
    lr: float
    use_energy: bool
    use_energy_propagation: bool
    use_classifier: bool
    use_energy_filter: bool
    upper_bound_id: float
    lower_bound_id: float
    lamda: float
    delta: float
    synthesis_ood: bool
    cov_mat: float
    sampling_ratio: float
    boundary_ratio: float
    boundary_sampling_ratio: float
    k: int
    T: float


def parser_parse_args(parser) -> Arguments:
    args = parser.parse_args()
    return Arguments(
        string=vars(args),
        dataset=args.dataset,
        ood_type=args.ood_type,
        data_dir=args.data_dir,
        device=args.device,
        cpu=args.cpu,
        seed=args.seed,
        epochs=args.epochs,
        method=args.method,
        backbone=args.backbone,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        lr=args.lr,
        use_energy=args.use_energy,
        use_energy_propagation=args.use_energy_propagation,
        use_classifier=args.use_classifier,
        use_energy_filter=args.use_energy_filter,
        upper_bound_id=args.upper_bound_id,
        lower_bound_id=args.lower_bound_id,
        lamda=args.lamda,
        delta=args.delta,
        synthesis_ood=args.synthesis_ood,
        cov_mat=args.cov_mat,
        sampling_ratio=args.sampling_ratio,
        boundary_ratio=args.boundary_ratio,
        boundary_sampling_ratio=args.boundary_sampling_ratio,
        k=args.k,
        T=args.T,
    )


def init_parser_args() -> Arguments:
    parser = argparse.ArgumentParser(description='General Training Pipeline')
    parser_add_main_args(parser)
    return parser_parse_args(parser)
