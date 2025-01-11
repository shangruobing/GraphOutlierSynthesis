## twitch

# Epoch: 79, Loss(↓): 0.6140, AUROC(↑): 65.51%, AUPR(↑): 70.55%, FPR95(↓): 91.41%, Accuracy(↑): 60.19%, Test Score(↑): 67.62%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "twitch" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -10

## arxiv

# Epoch: 95, Loss(↓): 1.4090, AUROC(↑): 60.63%, AUPR(↑): 62.32%, FPR95(↓): 92.94%, Accuracy(↑): 50.14%, Test Score(↑): 59.57%
python src/main.py --method "GNNOutlier" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "arxiv" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -10