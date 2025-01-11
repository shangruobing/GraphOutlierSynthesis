## twitch

# Epoch: 84, Loss(↓): 0.6070, AUROC(↑): 37.90%, AUPR(↑): 40.89%, FPR95(↓): 93.60%, Accuracy(↑): 49.39%, Test Score(↑): 66.32%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --dataset "twitch" --epochs 100 --lower_bound_id -3 --upper_bound_id -10 --num_layers 3

# Epoch: 57, Loss(↓): 0.5902, AUROC(↑): 55.80%, AUPR(↑): 55.88%, FPR95(↓): 91.75%, Accuracy(↑): 50.02%, Test Score(↑): 68.25%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --dataset "twitch" --epochs 100 --lower_bound_id -3 --upper_bound_id -10 --num_layers 3 \
  --use_energy

# Epoch: 56, Loss(↓): 0.5903, AUROC(↑): 77.23%, AUPR(↑): 78.59%, FPR95(↓): 71.16%, Accuracy(↑): 64.55%, Test Score(↑): 68.00%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --dataset "twitch" --epochs 100 --lower_bound_id -3 --upper_bound_id -10 --num_layers 3 \
  --use_energy --use_energy_propagation

# Epoch: 92, Loss(↓): 0.6432, AUROC(↑): 72.81%, AUPR(↑): 73.92%, FPR95(↓): 75.62%, Accuracy(↑): 64.95%, Test Score(↑): 62.91%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --dataset "twitch" --epochs 100 --lower_bound_id -3 --upper_bound_id -10 --num_layers 3 \
  --use_energy --use_energy_propagation --synthesis_ood

# Epoch: 96, Loss(↓): 0.6131, AUROC(↑): 83.10%, AUPR(↑): 85.37%, FPR95(↓): 64.88%, Accuracy(↑): 73.83%, Test Score(↑): 64.55%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --dataset "twitch" --epochs 100 --lower_bound_id -3 --upper_bound_id -10 --num_layers 3 \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood

# Epoch: 97, Loss(↓): 0.6408, AUROC(↑): 80.58%, AUPR(↑): 71.00%, FPR95(↓): 65.86%, Accuracy(↑): 76.62%, Test Score(↑): 62.36%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --dataset "twitch" --epochs 100 --lower_bound_id -3 --upper_bound_id -10 --num_layers 3 \
  --use_energy --use_energy_propagation --use_classifier --use_energy_filter --synthesis_ood

## arxiv

# Epoch: 99, Loss(↓): 1.4131, AUROC(↑): 52.12%, AUPR(↑): 53.57%, FPR95(↓): 95.87%, Accuracy(↑): 50.01%, Test Score(↑): 59.47%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --dataset "arxiv" --epochs 100 --lower_bound_id -2 --upper_bound_id -3 --num_layers 2

# Epoch: 99, Loss(↓): 1.5220, AUROC(↑): 49.49%, AUPR(↑): 51.95%, FPR95(↓): 96.79%, Accuracy(↑): 50.11%, Test Score(↑): 57.66%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --dataset "arxiv" --epochs 100 --lower_bound_id -2 --upper_bound_id -3 --num_layers 2 \
  --use_energy

# Epoch: 97, Loss(↓): 1.4498, AUROC(↑): 58.11%, AUPR(↑): 57.42%, FPR95(↓): 93.22%, Accuracy(↑): 50.29%, Test Score(↑): 59.16%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --dataset "arxiv" --epochs 100 --lower_bound_id -2 --upper_bound_id -3 --num_layers 2 \
  --use_energy --use_energy_propagation

# Epoch: 97, Loss(↓): 1.4261, AUROC(↑): 57.88%, AUPR(↑): 57.52%, FPR95(↓): 93.43%, Accuracy(↑): 56.02%, Test Score(↑): 59.15%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --dataset "arxiv" --epochs 100 --lower_bound_id -2 --upper_bound_id -3 --num_layers 2 \
  --use_energy --use_energy_propagation --synthesis_ood

# Epoch: 97, Loss(↓): 1.4411, AUROC(↑): 58.61%, AUPR(↑): 59.23%, FPR95(↓): 93.26%, Accuracy(↑): 51.60%, Test Score(↑): 58.94%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --dataset "arxiv" --epochs 100 --lower_bound_id -2 --upper_bound_id -3 --num_layers 2 \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood

# Epoch: 99, Loss(↓): 1.4333, AUROC(↑): 58.19%, AUPR(↑): 57.80%, FPR95(↓): 93.01%, Accuracy(↑): 54.34%, Test Score(↑): 58.90%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --dataset "arxiv" --epochs 100 --lower_bound_id -2 --upper_bound_id -3 --num_layers 2 \
  --use_energy --use_energy_propagation --use_classifier --use_energy_filter --synthesis_ood