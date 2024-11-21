## cora

# Epoch: 99, Loss(↓): 1.2208, AUROC(↑): 89.09%, AUPR(↑): 89.54%, FPR95(↓): 51.50%, Accuracy(↑): 81.65%, Test Score(↑): 75.90%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "cora" --ood_type "structure" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5

# Epoch: 99, Loss(↓): 1.1480, AUROC(↑): 90.79%, AUPR(↑): 91.91%, FPR95(↓): 51.40%, Accuracy(↑): 50.35%, Test Score(↑): 80.70%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "cora" --ood_type "feature" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5

# Epoch: 40, Loss(↓): 0.5399, AUROC(↑): 91.84%, AUPR(↑): 91.90%, FPR95(↓): 32.59%, Accuracy(↑): 54.59%, Test Score(↑): 89.56%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "cora" --ood_type "label" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5

## amazon-photo

# Epoch: 59, Loss(↓): 0.5716, AUROC(↑): 98.17%, AUPR(↑): 99.08%, FPR95(↓): 0.00%, Accuracy(↑): 76.27%, Test Score(↑): 89.91%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "amazon-photo" --ood_type "structure" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5

# Epoch: 87, Loss(↓): 0.4802, AUROC(↑): 98.43%, AUPR(↑): 98.88%, FPR95(↓): 0.68%, Accuracy(↑): 63.90%, Test Score(↑): 91.11%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "amazon-photo" --ood_type "feature" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5

# Epoch: 74, Loss(↓): 0.2711, AUROC(↑): 97.02%, AUPR(↑): 97.57%, FPR95(↓): 7.86%, Accuracy(↑): 85.57%, Test Score(↑): 96.28%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "amazon-photo" --ood_type "label" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5

## coauthor-cs

# Epoch: 60, Loss(↓): 0.6946, AUROC(↑): 82.79%, AUPR(↑): 88.82%, FPR95(↓): 97.14%, Accuracy(↑): 60.01%, Test Score(↑): 89.75%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "coauthor-cs" --ood_type "structure" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5

# Epoch: 95, Loss(↓): 0.7188, AUROC(↑): 83.80%, AUPR(↑): 89.02%, FPR95(↓): 93.80%, Accuracy(↑): 55.81%, Test Score(↑): 90.79%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "coauthor-cs" --ood_type "feature" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5

# Epoch: 85, Loss(↓): 0.4568, AUROC(↑): 85.01%, AUPR(↑): 87.33%, FPR95(↓): 61.48%, Accuracy(↑): 59.39%, Test Score(↑): 96.97%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "coauthor-cs" --ood_type "label" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5