## twitch

# Epoch: 90, Loss: 0.5108, AUROC: 52.22%, AUPR: 51.52%, FPR95: 92.13%, Accuracy: 50.00%, Test Score: 67.16%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset twitch --epochs 100 --lower_bound_id -3 --upper_bound_id -10 --num_layers 3

# Epoch: 56, Loss: 0.5743, AUROC: 55.80%, AUPR: 55.88%, FPR95: 91.49%, Accuracy: 50.00%, Test Score: 68.21%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset twitch --epochs 100 --lower_bound_id -3 --upper_bound_id -10 --num_layers 3 \
  --use_energy

# Epoch: 55, Loss: 0.5669, AUROC: 77.24%, AUPR: 78.61%, FPR95: 70.57%, Accuracy: 50.00%, Test Score: 68.04%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset twitch --epochs 100 --lower_bound_id -3 --upper_bound_id -10 --num_layers 3 \
  --use_energy --use_energy_propagation

# Epoch: 97, Loss: 0.5408, AUROC: 79.22%, AUPR: 81.05%, FPR95: 68.55%, Accuracy: 50.00%, Test Score: 65.39%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset twitch --epochs 100 --lower_bound_id -3 --upper_bound_id -10 --num_layers 3 \
  --use_energy --use_energy_propagation --synthesis_ood

# Epoch: 98, Loss: 0.5582, AUROC: 84.17%, AUPR: 85.86%, FPR95: 61.09%, Accuracy: 50.00%, Test Score: 62.99%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset twitch --epochs 100 --lower_bound_id -3 --upper_bound_id -10 --num_layers 3 \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood

# Epoch: 23, Loss: 0.7015, AUROC: 85.08%, AUPR: 86.47%, FPR95: 59.28%, Accuracy: 50.00%, Test Score: 63.71%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset twitch --epochs 100 --lower_bound_id -3 --upper_bound_id -10 --num_layers 3 \
  --use_energy --use_energy_propagation --use_classifier --use_energy_filter --synthesis_ood

## arxiv

# Epoch: 98, Loss: 1.3055, AUROC: 52.20%, AUPR: 53.63%, FPR95: 95.92%, Accuracy: 50.00%, Test Score: 59.74%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset arxiv --epochs 100 --lower_bound_id -2 --upper_bound_id -3 --num_layers 2

# Epoch: 99, Loss: 1.3609, AUROC: 56.77%, AUPR: 58.49%, FPR95: 94.71%, Accuracy: 49.92%, Test Score: 59.76%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset arxiv --epochs 100 --lower_bound_id -2 --upper_bound_id -3 --num_layers 2 \
  --use_energy

# Epoch: 89, Loss: 1.3355, AUROC: 60.88%, AUPR: 62.68%, FPR95: 92.98%, Accuracy: 51.74%, Test Score: 60.32%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset arxiv --epochs 100 --lower_bound_id -2 --upper_bound_id -3 --num_layers 2 \
  --use_energy --use_energy_propagation

# Epoch: 82, Loss: 1.3384, AUROC: 60.80%, AUPR: 62.56%, FPR95: 92.85%, Accuracy: 52.05%, Test Score: 59.95%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset arxiv --epochs 100 --lower_bound_id -2 --upper_bound_id -3 --num_layers 2 \
  --use_energy --use_energy_propagation --synthesis_ood

# Epoch: 99, Loss: 1.7965, AUROC: 60.71%, AUPR: 62.41%, FPR95: 92.92%, Accuracy: 51.50%, Test Score: 59.13%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset arxiv --epochs 100 --lower_bound_id -2 --upper_bound_id -3 --num_layers 2 \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood

# Epoch: 98, Loss: 1.9732, AUROC: 58.34%, AUPR: 58.29%, FPR95: 93.06%, Accuracy: 49.96%, Test Score: 58.77%
python main.py --method gnnsafe --backbone gcn --device 0 --dataset arxiv --epochs 100 --lower_bound_id -2 --upper_bound_id -3 --num_layers 2 \
  --use_energy --use_energy_propagation --use_classifier --use_energy_filter --synthesis_ood