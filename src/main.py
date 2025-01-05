import sys
from os.path import dirname, abspath
from pprint import pprint

import torch
import torch.nn as nn
from torch_geometric.data import Data

BASE_DIR = dirname(dirname(abspath(__file__)))
print(f"BASE_DIR:{BASE_DIR}")
sys.path.append(BASE_DIR)

from src.common.logger import DetectLogger
from src.common.parse import init_parser_args, Arguments
from src.common.utils import get_device, fix_seed, get_now_datetime
from src.common.recorder import Recorder
from src.model.baselines import MSP, OE, ODIN, Mahalanobis, MaxLogits, EnergyModel, EnergyProp, GNNSafe
from src.model.data_utils import evaluate_detect, eval_acc
from src.model.dataset import load_dataset, create_knn_dataset


def setup_args() -> tuple[Arguments, torch.device]:
    args = init_parser_args()
    pprint(args.string)
    fix_seed(args.seed)
    device = get_device(device=args.device, cpu=args.cpu)
    return args, device


def setup_dataset(args: Arguments, device: torch.device):
    print(f"\n{'Begin Prepare Dataset':=^80}")

    dataset_ind, dataset_ood_tr, dataset_ood_te = load_dataset(args)

    num_classes = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
    num_features = dataset_ind.num_features

    if args.synthesis_ood:
        synthesis_ood_dataset = create_knn_dataset(
            data=dataset_ind,
            cov_mat=args.cov_mat,
            sampling_ratio=args.sampling_ratio,
            boundary_ratio=args.boundary_ratio,
            boundary_sampling_ratio=args.boundary_sampling_ratio,
            k=args.k,
            device=device
        ).to(device=device)
    else:
        synthesis_ood_dataset = Data()
        synthesis_ood_dataset.num_nodes = 0

    print(f"ind train dataset {args.dataset}: nodes {dataset_ind.num_nodes} | edges {dataset_ind.num_edges} | classes {num_classes} | features {num_features}")
    print(f"ood train dataset {args.dataset}: nodes {dataset_ood_tr.num_nodes} | edges {dataset_ood_tr.num_edges}")
    print(f"ood test  dataset {args.dataset}: nodes {dataset_ood_te.num_nodes} | edges {dataset_ood_te.num_edges}")
    print(f"synthesis dataset {args.dataset}: nodes {synthesis_ood_dataset.num_nodes} | edges {synthesis_ood_dataset.num_edges}")

    dataset_ind.to(device=device)
    dataset_ood_tr.to(device=device)
    dataset_ood_te.to(device=device)

    print(f"{'End Prepare Dataset':=^80}")

    return dataset_ind, dataset_ood_tr, dataset_ood_te, synthesis_ood_dataset, num_classes, num_features


def setup_model(args: Arguments, num_features: int, num_classes: int, device: torch.device):
    if args.method == 'msp':
        model = MSP(num_features, num_classes, args)
    elif args.method == 'gnnsafe':
        model = GNNSafe(num_features, num_classes, args)
    elif args.method == 'OE':
        model = OE(num_features, num_classes, args)
    elif args.method == "ODIN":
        model = ODIN(num_features, num_classes, args)
    elif args.method == "Mahalanobis":
        model = Mahalanobis(num_features, num_classes, args)
    elif args.method == 'MaxLogits':
        model = MaxLogits(num_features, num_classes, args)
    elif args.method == 'EnergyModel':
        model = EnergyModel(num_features, num_classes, args)
    elif args.method == 'EnergyProp':
        model = EnergyProp(num_features, num_classes, args)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    print(model)

    model.train()
    model.reset_parameters()
    model.to(device)

    criterion = nn.NLLLoss()

    optimizer = torch.optim.Adam(
        params=[
            {'params': model.encoder.parameters(), 'lr': args.lr},
            {'params': model.classifier.parameters(), 'lr': args.lr},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    logger = DetectLogger()

    return model, criterion, optimizer, logger


def train():
    print(f"\n{'Begin Time: ' + get_now_datetime():=^80}")
    args, device = setup_args()
    dataset_ind, dataset_ood_tr, dataset_ood_te, synthesis_ood_dataset, num_classes, num_features = setup_dataset(args=args, device=device)
    model, criterion, optimizer, logger = setup_model(args=args, num_features=num_features, num_classes=num_classes, device=device)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.loss_compute(dataset_ind, dataset_ood_tr, synthesis_ood_dataset, criterion, device, args)
        loss.backward()
        optimizer.step()
        metric = evaluate_detect(model, dataset_ind, dataset_ood_te, criterion, eval_acc, args, device)
        logger.log(epoch, loss, *metric)

    print(f"\n{'Final Statistics':=^80}")
    metrics = logger.get_statistics()

    Recorder.insert_row(
        args=args,
        model=str(model),
        epoch_info=logger.epoch_info,
        auroc=metrics.auroc,
        aupr=metrics.aupr,
        fpr=metrics.fpr,
        accuracy=metrics.accuracy,
        score=metrics.score
    )
    print(f"\n{'End Time: ' + get_now_datetime():=^80}")


if __name__ == '__main__':
    train()
