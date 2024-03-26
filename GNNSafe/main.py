import argparse
import sys
from pprint import pprint
import torch
import torch.nn as nn

sys.path.append('..')

from baselines import MSP, OE, ODIN, Mahalanobis, MaxLogits, EnergyModel, EnergyProp, GNNSafe
from data_utils import evaluate_classify, evaluate_detect, eval_acc, eval_rocauc, rand_splits
from dataset import load_dataset
from logger import ClassifyLogger, DetectLogger
from parse import parser_add_main_args

from OutliersGenerate.utils import get_device, fix_seed
from OutliersGenerate.recorder import insert_row

parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
pprint(vars(args))

fix_seed(args.seed)
device = get_device(args)

dataset_ind, dataset_ood_te = load_dataset(args)

if len(dataset_ind.y.shape) == 1:
    dataset_ind.y = dataset_ind.y.unsqueeze(1)
# if isinstance(dataset_ood_te, list):
#     for data in dataset_ood_te:
#         if len(data.y.shape) == 1:
#             data.y = data.y.unsqueeze(1)
# else:
#     if len(dataset_ood_te.y.shape) == 1:
#         dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)
if len(dataset_ood_te.y.shape) == 1:
    dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)

if hasattr(dataset_ind, "train_mask"):
    print("Using provided splits")
else:
    print("Using random splits")
    train_mask, val_mask, test_mask = rand_splits(dataset_ind.num_nodes)
    dataset_ind.train_mask = train_mask
    dataset_ind.val_mask = val_mask
    dataset_ind.test_mask = test_mask

num_classes = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
num_features = dataset_ind.num_features

print(f"ind train dataset {args.dataset}: nodes {dataset_ind.num_nodes} | edges {dataset_ind.num_edges} | classes {num_classes} | features {num_features}")
print(f"ood test  dataset {args.dataset}: nodes {dataset_ood_te.num_nodes} | edges {dataset_ood_te.num_edges}")

if args.method == 'msp':
    model = MSP(num_features, num_classes, args)
elif args.method in 'gnnsafe':
    model = GNNSafe(num_features, num_classes, args)
elif args.method == 'OE':
    model = OE(num_features, num_classes, args)
elif args.method == "ODIN":
    model = ODIN(num_features, num_classes, args)
elif args.method == "Mahalanobis":
    model = Mahalanobis(num_features, num_classes, args)
elif args.method == 'maxlogits':
    model = MaxLogits(num_features, num_classes, args)
elif args.method == 'energymodel':
    model = EnergyModel(num_features, num_classes, args)
elif args.method == 'energyprop':
    model = EnergyProp(num_features, num_classes, args)
else:
    raise ValueError(f"Unknown method: {args.method}")

if args.dataset in ('proteins', 'ppi'):  # multi-label binary classification
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

if args.dataset in ('proteins', 'ppi', 'twitch'):  # binary classification
    eval_func = eval_rocauc
else:
    eval_func = eval_acc

if args.mode == 'classify':
    logger = ClassifyLogger()
else:
    logger = DetectLogger()

model.train()
# print(model)

epoch_info = ""
model.reset_parameters()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# optimizer = torch.optim.Adam(model.encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# classifier_optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr * 10, weight_decay=args.weight_decay)

for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()
    # classifier_optimizer.zero_grad()
    loss = model.loss_compute(dataset_ind, criterion, device, args)
    # loss = model.loss_compute(dataset_ind, dataset_ood_tr, criterion, device, args)
    loss.backward()
    # classifier_optimizer.step()
    optimizer.step()

    if args.mode == 'classify':
        result = evaluate_classify(model, dataset_ind, eval_func, criterion, args, device)
        logger.add_result(result)

        if epoch % args.display_step == 0:
            info = f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * result[0]:.2f}%, Valid: {100 * result[1]:.2f}%, Test: {100 * result[2]:.2f}%'
            epoch_info += info + '\n'
            print(info)
    else:
        result = evaluate_detect(model, dataset_ind, dataset_ood_te, criterion, eval_func, args, device)
        logger.add_result(result)

        if epoch % args.display_step == 0:
            info = f'Epoch: {epoch:02d}, Loss: {loss:.4f}, AUROC: {100 * result[0]:.2f}%, AUPR: {100 * result[1]:.2f}%, FPR95: {100 * result[2]:.2f}%, Test Score: {100 * result[-2]:.2f}%'
            epoch_info += info + '\n'
            print(info)

metrics = logger.get_statistics()
insert_row(
    args=args,
    model=str(model),
    epoch_info=epoch_info,
    AUROC=metrics.get("AUROC"),
    AUPR=metrics.get("AUPR"),
    FPR=metrics.get("FPR"),
    SCORE=metrics.get("SCORE")
)
