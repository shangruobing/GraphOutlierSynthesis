## cora

# Epoch: 99, Loss: 0.0047, AUROC: 88.48%, AUPR: 88.77%, FPR95: 53.00%, Accuracy: 51.85%, Test Score: 74.30%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "cora" --ood_type "structure" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5

# Epoch: 99, Loss: 0.0040, AUROC: 89.69%, AUPR: 90.97%, FPR95: 60.40%, Accuracy: 50.00%, Test Score: 80.70%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "cora" --ood_type "feature" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5

# Epoch: 99, Loss: 0.0020, AUROC: 91.28%, AUPR: 91.65%, FPR95: 41.77%, Accuracy: 50.00%, Test Score: 89.24%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "cora" --ood_type "label" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5

## amazon-photo

# Epoch: 99, Loss: 0.1826, AUROC: 97.97%, AUPR: 98.92%, FPR95: 0.00%, Accuracy: 49.92%, Test Score: 89.81%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "amazon-photo" --ood_type "structure" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5

# Epoch: 99, Loss: 0.1610, AUROC: 97.98%, AUPR: 98.52%, FPR95: 0.84%, Accuracy: 50.00%, Test Score: 88.24%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "amazon-photo" --ood_type "feature" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5

# Epoch: 99, Loss: 0.0855, AUROC: 96.11%, AUPR: 96.53%, FPR95: 10.34%, Accuracy: 50.41%, Test Score: 96.07%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "amazon-photo" --ood_type "label" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5

## coauthor-cs

# Epoch: 99, Loss: 0.4108, AUROC: 85.05%, AUPR: 90.36%, FPR95: 97.36%, Accuracy: 50.01%, Test Score: 89.97%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "coauthor-cs" --ood_type "structure" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5

# Epoch: 47, Loss: 0.0938, AUROC: 87.86%, AUPR: 92.57%, FPR95: 97.75%, Accuracy: 50.00%, Test Score: 91.64%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "coauthor-cs" --ood_type "feature" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5

# Epoch: 27, Loss: 0.0606, AUROC: 90.02%, AUPR: 90.64%, FPR95: 44.86%, Accuracy: 50.00%, Test Score: 98.26%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "coauthor-cs" --ood_type "label" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5