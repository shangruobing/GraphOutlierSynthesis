import argparse
from dataclasses import dataclass
from src.common.config import DATASET_PATH


def parser_add_main_args(parser):
    # setup and protocol
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--ood_type', type=str, default='structure', choices=['structure', 'label', 'feature', 'knn'])
    parser.add_argument('--data_dir', type=str, default=str(DATASET_PATH))
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)

    # model network
    parser.add_argument('--method', type=str, default='msp')
    parser.add_argument('--backbone', type=str, default='gcn')
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
        use_energy_filter=args.use_energy_filter
    )


def init_parser_args() -> Arguments:
    parser = argparse.ArgumentParser(description='General Training Pipeline')
    parser_add_main_args(parser)
    return parser_parse_args(parser)
