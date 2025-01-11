# Epoch: 99, Loss(↓): 0.7892, AUROC(↑): 75.21%, AUPR(↑): 72.05%, FPR95(↓): 80.10%, Accuracy(↑): 69.60%, Test Score(↑): 78.50%
python src/main.py --method "MSP" --backbone "gcn" --device 0 --epochs 100 --dataset "cora" --ood_type "structure"

# Epoch: 66, Loss(↓): 1.1002, AUROC(↑): 69.60%, AUPR(↑): 69.39%, FPR95(↓): 92.40%, Accuracy(↑): 49.80%, Test Score(↑): 71.40%
python src/main.py --method "OE" --backbone "gcn" --device 0 --epochs 100 --dataset "cora" --ood_type "structure"

# Epoch: 99, Loss(↓): 0.7893, AUROC(↑): 41.14%, AUPR(↑): 44.93%, FPR95(↓): 99.60%, Accuracy(↑): 50.00%, Test Score(↑): 78.50%
python src/main.py --method "Mahalanobis" --backbone "gcn" --device 0 --epochs 100 --dataset "cora" --ood_type "structure"

# Epoch: 99, Loss(↓): 0.7888, AUROC(↑): 74.50%, AUPR(↑): 70.42%, FPR95(↓): 78.70%, Accuracy(↑): 49.90%, Test Score(↑): 78.60%
python src/main.py --method "MaxLogits" --backbone "gcn" --device 0 --epochs 100 --dataset "cora" --ood_type "structure"

# Epoch: 99, Train Loss: 0.0038, Val Loss: 0.7888, AUROC: 73.50%, AUPR: 69.75%, FPR95: 83.60%, Accuracy: 49.90%, Test Score: 78.60%
python src/main.py --method "EnergyModel" --backbone "gcn" --device 0 --epochs 100 --dataset "cora" --ood_type "structure"

# Epoch: 99, Loss(↓): 0.7887, AUROC(↑): 85.17%, AUPR(↑): 85.12%, FPR95(↓): 72.70%, Accuracy(↑): 51.60%, Test Score(↑): 78.50%
python src/main.py --method "EnergyProp" --backbone "gcn" --device 0 --epochs 100 --dataset "cora" --ood_type "structure"

# Epoch: 48, Loss(↓): 0.7774, AUROC(↑): 83.38%, AUPR(↑): 72.84%, FPR95(↓): 91.29%, Accuracy(↑): 80.99%, Test Score(↑): 78.70%
python src/main.py --method "GNNSafe" --backbone "gcn" --device 0 --epochs 100 --dataset "cora" --ood_type "structure"

# Epoch: 42, Loss(↓): 1.3120, AUROC(↑): 98.12%, AUPR(↑): 98.46%, FPR95(↓): 3.40%, Accuracy(↑): 93.05%, Test Score(↑): 59.10%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --epochs 100 --dataset "cora" --lower_bound_id -3 --upper_bound_id -10 --num_layers 3  --use_energy --use_energy_propagation --use_classifier --synthesis_ood