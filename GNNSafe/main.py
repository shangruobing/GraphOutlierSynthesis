import argparse
import sys
from pprint import pprint

sys.path.append('..')

from baselines import *
from data_utils import evaluate_classify, evaluate_detect, eval_acc, eval_rocauc, rand_splits
from dataset import load_dataset
from gnnsafe import *
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

dataset_ind, dataset_ood_tr, dataset_ood_te = load_dataset(args)
"""
Actor Datasets:
dataset_ind:
Data(x=[7600, 932], edge_index=[2, 30019], y=[7600], 
    train_mask=[7600, 10], val_mask=[7600, 10], test_mask=[7600, 10], 
    node_idx=[7600])

dataset_ood_tr
Data(x=[7600, 932], edge_index=[2, 21146], y=[7600], node_idx=[7600])

dataset_ood_te
Data(x=[7600, 932], edge_index=[2, 21106], y=[7600], node_idx=[7600])
"""

if len(dataset_ind.y.shape) == 1:
    dataset_ind.y = dataset_ind.y.unsqueeze(1)
if len(dataset_ood_tr.y.shape) == 1:
    dataset_ood_tr.y = dataset_ood_tr.y.unsqueeze(1)
if isinstance(dataset_ood_te, list):
    for data in dataset_ood_te:
        if len(data.y.shape) == 1:
            data.y = data.y.unsqueeze(1)
else:
    if len(dataset_ood_te.y.shape) == 1:
        dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)

if args.dataset in ['cora', 'citeseer', 'pubmed']:
    pass
else:
    dataset_ind.splits = rand_splits(dataset_ind.node_idx, train_prop=args.train_prop, valid_prop=args.valid_prop)

c = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
d = dataset_ind.x.shape[1]
"""
Actor Datasets:
# d c 932 5
# features 932 | classes 5
"""

print(
    f"ind dataset {args.dataset}: all nodes {dataset_ind.num_nodes} | centered nodes {dataset_ind.node_idx.shape[0]} | edges {dataset_ind.edge_index.size(1)} | "
    + f"classes {c} | feats {d}")
print(
    f"ood tr dataset {args.dataset}: all nodes {dataset_ood_tr.num_nodes} | centered nodes {dataset_ood_tr.node_idx.shape[0]} | edges {dataset_ood_tr.edge_index.size(1)}")

if isinstance(dataset_ood_te, list):
    for i, data in enumerate(dataset_ood_te):
        print(
            f"ood te dataset {i} {args.dataset}: all nodes {data.num_nodes} | centered nodes {data.node_idx.shape[0]} | edges {data.edge_index.size(1)}")
else:
    print(
        f"ood te dataset {args.dataset}: all nodes {dataset_ood_te.num_nodes} | centered nodes {dataset_ood_te.node_idx.shape[0]} | edges {dataset_ood_te.edge_index.size(1)}")

if args.method == 'msp':
    model = MSP(d, c, args).to(device)
elif args.method in 'gnnsafe':
    model = GNNSafe(d, c, args).to(device)
elif args.method == 'OE':
    model = OE(d, c, args).to(device)
elif args.method == "ODIN":
    model = ODIN(d, c, args).to(device)
elif args.method == "Mahalanobis":
    model = Mahalanobis(d, c, args).to(device)
else:
    raise ValueError(f"Unknown method: {args.method}")

if args.dataset in ('proteins', 'ppi'):  # multi-label binary classification
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

# metric for classification #
if args.dataset in ('proteins', 'ppi', 'twitch'):  # binary classification
    eval_func = eval_rocauc
else:
    eval_func = eval_acc

# logger for result report #
if args.mode == 'classify':
    logger = ClassifyLogger(args.runs, args)
else:
    logger = DetectLogger(args.runs, args)

model.train()
print('MODEL:', model)

epoch_info = ""
for run in range(args.runs):
    model.reset_parameters()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.loss_compute(dataset_ind, dataset_ood_tr, criterion, device, args)
        loss.backward()
        optimizer.step()

        if args.mode == 'classify':
            result = evaluate_classify(model, dataset_ind, eval_func, criterion, args, device)
            logger.add_result(run, result)

            if epoch % args.display_step == 0:
                info = f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * result[0]:.2f}%, Valid: {100 * result[1]:.2f}%, Test: {100 * result[2]:.2f}%'
                epoch_info += info + '\n'
                print(info)
        else:
            result = evaluate_detect(model, dataset_ind, dataset_ood_te, criterion, eval_func, args, device)
            logger.add_result(run, result)

            if epoch % args.display_step == 0:
                info = f'Epoch: {epoch:02d}, Loss: {loss:.4f}, AUROC: {100 * result[0]:.2f}%, AUPR: {100 * result[1]:.2f}%, FPR95: {100 * result[2]:.2f}%, Test Score: {100 * result[-2]:.2f}%'
                epoch_info += info + '\n'
                print(info)
    logger.print_statistics(run)

results = logger.print_statistics()

# Save results #
# if args.mode == 'detect':
#     save_result(results, args)


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
