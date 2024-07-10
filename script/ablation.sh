## twitch

# Epoch: 96, Loss: 0.5545, AUROC: 61.76%, AUPR: 59.62%, FPR95: 89.09%, Accuracy: 57.83%, Test Score: 66.23%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset twitch --epochs 100 --lower_bound_id -1 --upper_bound_id -10

# Epoch: 79, Loss: 0.5570, AUROC: 50.75%, AUPR: 51.91%, FPR95: 94.91%, Accuracy: 50.00%, Test Score: 68.38%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset twitch --epochs 100 --lower_bound_id -1 --upper_bound_id -10 \
  --use_energy

# Epoch: 79, Loss: 0.5550, AUROC: 56.86%, AUPR: 56.54%, FPR95: 93.56%, Accuracy: 50.00%, Test Score: 68.04%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset twitch --epochs 100 --lower_bound_id -1 --upper_bound_id -10 \
  --use_energy --use_energy_propagation

# Epoch: 30, Loss: 0.5751, AUROC: 61.48%, AUPR: 65.45%, FPR95: 92.29%, Accuracy: 50.00%, Test Score: 64.51%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset twitch --epochs 100 --lower_bound_id -1 --upper_bound_id -10 \
  --use_energy --use_energy_propagation --synthesis_ood

# Epoch: 91, Loss: 0.5671, AUROC: 65.39%, AUPR: 70.26%, FPR95: 91.24%, Accuracy: 50.00%, Test Score: 66.99%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset twitch --epochs 100 --lower_bound_id -1 --upper_bound_id -10 \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood

# Epoch: 26, Loss: 0.5759, AUROC: 63.92%, AUPR: 68.31%, FPR95: 92.04%, Accuracy: 50.00%, Test Score: 64.34%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset twitch --epochs 100 --lower_bound_id -1 --upper_bound_id -10 \
  --use_energy --use_energy_propagation --use_classifier --use_energy_filter --synthesis_ood

## arxiv

# Epoch: 98, Loss: 1.3055, AUROC: 52.20%, AUPR: 53.63%, FPR95: 95.92%, Accuracy: 50.00%, Test Score: 59.74%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset arxiv --epochs 100 --lower_bound_id -1 --upper_bound_id -10

# Epoch: 99, Loss: 1.3609, AUROC: 56.77%, AUPR: 58.49%, FPR95: 94.71%, Accuracy: 49.92%, Test Score: 59.76%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset arxiv --epochs 100 --lower_bound_id -1 --upper_bound_id -10 \
  --use_energy

# Epoch: 89, Loss: 1.3355, AUROC: 60.88%, AUPR: 62.68%, FPR95: 92.98%, Accuracy: 51.74%, Test Score: 60.32%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset arxiv --epochs 100 --lower_bound_id -1 --upper_bound_id -10 \
  --use_energy --use_energy_propagation

# Epoch: 82, Loss: 1.3384, AUROC: 60.80%, AUPR: 62.56%, FPR95: 92.85%, Accuracy: 52.05%, Test Score: 59.95%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset arxiv --epochs 100 --lower_bound_id -1 --upper_bound_id -10 \
  --use_energy --use_energy_propagation --synthesis_ood

# Epoch: 99, Loss: 1.7965, AUROC: 60.71%, AUPR: 62.41%, FPR95: 92.92%, Accuracy: 51.50%, Test Score: 59.13%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset arxiv --epochs 100 --lower_bound_id -1 --upper_bound_id -10 \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood

# Epoch: 98, Loss: 1.9732, AUROC: 58.34%, AUPR: 58.29%, FPR95: 93.06%, Accuracy: 49.96%, Test Score: 58.77%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset arxiv --epochs 100 --lower_bound_id -2 --upper_bound_id -3 \
  --use_energy --use_energy_propagation --use_classifier --use_energy_filter --synthesis_ood