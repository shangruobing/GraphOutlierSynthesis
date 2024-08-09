# ==================== Begin Generate Outliers ====================
# How many OOD samples to generate: 2708
# How many ID samples are defined as points near the boundary: 270
# How many boundary used to generate outliers: 135
# The number of nearest neighbors to return: 100
# The generate outliers device is: cuda
# The generate outliers time is 1.65s
# ===================== End Generate Outliers =====================
# Epoch: 99, Loss: 0.0043, AUROC: 88.50%, AUPR: 89.01%, FPR95: 55.80%, Accuracy: 51.95%, Test Score: 74.80%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "cora" --ood_type "structure" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5 \
  --cov_mat 0.1 --sampling_ratio 1.0 --boundary_ratio 0.1 --boundary_sampling_ratio 0.5 --k 100


# ==================== Begin Generate Outliers ====================
# How many OOD samples to generate: 2708
# How many ID samples are defined as points near the boundary: 270
# How many boundary used to generate outliers: 135
# The number of nearest neighbors to return: 10
# The generate outliers device is: cuda
# The generate outliers time is 1.45s
# ===================== End Generate Outliers =====================
# Epoch: 99, Loss: 0.0042, AUROC: 90.23%, AUPR: 91.15%, FPR95: 50.20%, Accuracy: 52.15%, Test Score: 76.90%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "cora" --ood_type "structure" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5 \
  --cov_mat 0.1 --sampling_ratio 1.0 --boundary_ratio 0.1 --boundary_sampling_ratio 0.5 --k 10


# ==================== Begin Generate Outliers ====================
# How many OOD samples to generate: 1624
# How many ID samples are defined as points near the boundary: 270
# How many boundary used to generate outliers: 135
# The number of nearest neighbors to return: 10
# The generate outliers device is: cuda
# The generate outliers time is 1.2s
# ===================== End Generate Outliers =====================
# Epoch: 31, Loss: 0.0355, AUROC: 88.31%, AUPR: 89.62%, FPR95: 61.50%, Accuracy: 51.95%, Test Score: 80.80%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "cora" --ood_type "structure" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5 \
  --cov_mat 0.1 --sampling_ratio 0.6 --boundary_ratio 0.1 --boundary_sampling_ratio 0.5 --k 10