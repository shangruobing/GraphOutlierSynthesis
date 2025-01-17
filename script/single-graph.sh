## cora

#Epoch: 99, Train Loss: 0.3757, Val Loss: 1.5004, AUROC: 80.82%, AUPR: 78.65%, FPR95: 66.43%, Accuracy: 63.88%, Test Score: 70.20%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "cora" --ood_type "structure" --use_classifier --synthesis_ood

# Epoch: 98, Train Loss: 0.3709, Val Loss: 1.2207, AUROC: 82.56%, AUPR: 81.83%, FPR95: 64.48%, Accuracy: 71.23%, Test Score: 74.00%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "cora" --ood_type "feature" --use_classifier --synthesis_ood

# Epoch: 40, Loss(↓): 0.5399, AUROC(↑): 91.84%, AUPR(↑): 91.90%, FPR95(↓): 32.59%, Accuracy(↑): 54.59%, Test Score(↑): 89.56%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "cora" --ood_type "label" --use_classifier --synthesis_ood

## amazon-photo

# Epoch: 59, Loss(↓): 0.5716, AUROC(↑): 98.17%, AUPR(↑): 99.08%, FPR95(↓): 0.00%, Accuracy(↑): 76.27%, Test Score(↑): 89.91%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "amazon-photo" --ood_type "structure" --use_classifier --synthesis_ood

# Epoch: 87, Loss(↓): 0.4802, AUROC(↑): 98.43%, AUPR(↑): 98.88%, FPR95(↓): 0.68%, Accuracy(↑): 63.90%, Test Score(↑): 91.11%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "amazon-photo" --ood_type "feature" --use_classifier --synthesis_ood
# Epoch: 74, Loss(↓): 0.2711, AUROC(↑): 97.02%, AUPR(↑): 97.57%, FPR95(↓): 7.86%, Accuracy(↑): 85.57%, Test Score(↑): 96.28%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "amazon-photo" --ood_type "label" --use_classifier --synthesis_ood

## coauthor-cs

# Epoch: 60, Loss(↓): 0.6946, AUROC(↑): 82.79%, AUPR(↑): 88.82%, FPR95(↓): 97.14%, Accuracy(↑): 60.01%, Test Score(↑): 89.75%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "coauthor-cs" --ood_type "structure" --use_classifier --synthesis_ood

# Epoch: 95, Loss(↓): 0.7188, AUROC(↑): 83.80%, AUPR(↑): 89.02%, FPR95(↓): 93.80%, Accuracy(↑): 55.81%, Test Score(↑): 90.79%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "coauthor-cs" --ood_type "feature" --use_classifier --synthesis_ood

# Epoch: 85, Loss(↓): 0.4568, AUROC(↑): 85.01%, AUPR(↑): 87.33%, FPR95(↓): 61.48%, Accuracy(↑): 59.39%, Test Score(↑): 96.97%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "coauthor-cs" --ood_type "label" --use_classifier --synthesis_ood