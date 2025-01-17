#Epoch: 99, Train Loss: 0.1922, Val Loss: 1.0137, AUROC: 85.17%, AUPR: 80.41%, FPR95: 81.46%, Accuracy: 86.06%, Test Score: 75.90%
python src/main.py --method "MSP" --backbone "gcn" --device 0 --epochs 100 --dataset "cora" --ood_type "structure"

#Epoch: 99, Train Loss: 1.2654, Val Loss: 1.0605, AUROC: 83.55%, AUPR: 78.99%, FPR95: 82.57%, Accuracy: 73.27%, Test Score: 74.20%
python src/main.py --method "OE" --backbone "gcn" --device 0 --epochs 100 --dataset "cora" --ood_type "structure"

#Epoch: 99, Train Loss: 0.1927, Val Loss: 1.0149, AUROC: 60.89%, AUPR: 41.02%, FPR95: 94.35%, Accuracy: 73.06%, Test Score: 76.00%
python src/main.py --method "Mahalanobis" --backbone "gcn" --device 0 --epochs 100 --dataset "cora" --ood_type "structure"

#Epoch: 99, Train Loss: 0.1923, Val Loss: 1.0141, AUROC: 83.87%, AUPR: 79.21%, FPR95: 80.91%, Accuracy: 73.38%, Test Score: 76.00%
python src/main.py --method "MaxLogits" --backbone "gcn" --device 0 --epochs 100 --dataset "cora" --ood_type "structure"

#Epoch: 99, Train Loss: 0.1918, Val Loss: 1.0145, AUROC: 82.10%, AUPR: 77.47%, FPR95: 78.40%, Accuracy: 73.35%, Test Score: 75.80%
python src/main.py --method "EnergyModel" --backbone "gcn" --device 0 --epochs 100 --dataset "cora" --ood_type "structure"

#Epoch: 99, Train Loss: 0.1915, Val Loss: 1.0134, AUROC: 90.08%, AUPR: 89.71%, FPR95: 85.34%, Accuracy: 75.11%, Test Score: 75.80%
python src/main.py --method "EnergyProp" --backbone "gcn" --device 0 --epochs 100 --dataset "cora" --ood_type "structure"

#Epoch: 99, Train Loss: 1.9117, Val Loss: 1.6741, AUROC: 97.07%, AUPR: 95.53%, FPR95: 17.06%, Accuracy: 75.81%, Test Score: 63.20%
python src/main.py --method "GNNSafe" --backbone "gcn" --device 0 --epochs 100 --dataset "cora" --ood_type "structure"

# Epoch: 99, Train Loss: 0.0000, Val Loss: 1.4735, AUROC: 92.29%, AUPR: 87.02%, FPR95: 46.57%, Accuracy: 81.34%, Test Score: 69.70%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --epochs 100 --dataset "cora" --ood_type "structure" --use_classifier

# Epoch: 99, Train Loss: 0.0024, Val Loss: 1.2380, AUROC: 94.87%, AUPR: 92.83%, FPR95: 33.20%, Accuracy: 84.36%, Test Score: 68.50%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --epochs 100 --dataset "cora" --ood_type "structure" --use_classifier --synthesis_ood