#  ============================Begin Generate Outliers=============================
#  Number of OOD samples to generate: 2708
#  Number of ID samples defined as points near the boundary: 270
#  Number of boundaries used to generate outliers: 135
#  Number of nearest neighbors to return: 100
#  Device for generating outliers: cuda
#  Time taken to generate outliers 1.94s
#  =============================End Generate Outliers==============================
# Epoch: 99, Loss(↓): 1.2128, AUROC(↑): 88.96%, AUPR(↑): 89.45%, FPR95(↓): 54.10%, Accuracy(↑): 80.55%, Test Score(↑): 76.60%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "cora" --ood_type "structure" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5 \
  --cov_mat 0.1 --sampling_ratio 1.0 --boundary_ratio 0.1 --boundary_sampling_ratio 0.5 --k 100


#  ============================Begin Generate Outliers=============================
#  Number of OOD samples to generate: 2708
#  Number of ID samples defined as points near the boundary: 270
#  Number of boundaries used to generate outliers: 135
#  Number of nearest neighbors to return: 10
#  Device for generating outliers: cuda
#  Time taken to generate outliers 1.54s
#  =============================End Generate Outliers==============================
# Epoch: 35, Loss(↓): 1.1887, AUROC(↑): 88.97%, AUPR(↑): 90.22%, FPR95(↓): 59.40%, Accuracy(↑): 73.95%, Test Score(↑): 79.50%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "cora" --ood_type "structure" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5 \
  --cov_mat 0.1 --sampling_ratio 1.0 --boundary_ratio 0.1 --boundary_sampling_ratio 0.5 --k 10


#  ============================Begin Generate Outliers=============================
#  Number of OOD samples to generate: 1624
#  Number of ID samples defined as points near the boundary: 270
#  Number of boundaries used to generate outliers: 135
#  Number of nearest neighbors to return: 10
#  Device for generating outliers: cuda
#  Time taken to generate outliers 1.42s
#  =============================End Generate Outliers==============================
# Epoch: 31, Loss(↓): 1.1330, AUROC(↑): 88.14%, AUPR(↑): 89.48%, FPR95(↓): 62.40%, Accuracy(↑): 60.70%, Test Score(↑): 80.50%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "cora" --ood_type "structure" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -5 \
  --cov_mat 0.1 --sampling_ratio 0.6 --boundary_ratio 0.1 --boundary_sampling_ratio 0.5 --k 10