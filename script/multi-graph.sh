## twitch

# Epoch: 90, Loss: 0.5641, AUROC: 67.71%, AUPR: 73.19%, FPR95: 90.65%, Accuracy: 50.00%, Test Score: 65.68%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "twitch" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -10

## arxiv

# Epoch: 99, Loss: 1.7960, AUROC: 60.74%, AUPR: 62.46%, FPR95: 92.88%, Accuracy: 51.59%, Test Score: 59.07%
python main.py --method "gnnsafe" --backbone "gcn" --device 0 --epochs 100 \
  --dataset "arxiv" \
  --use_energy --use_energy_propagation --use_classifier --synthesis_ood \
  --lower_bound_id -1 --upper_bound_id -10