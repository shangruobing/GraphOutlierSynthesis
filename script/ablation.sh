cd ../src || exit

echo "Parameters:"

echo -e "\nStart training...\n"

python main.py --method gnnsafe --backbone gcn --device 0 --dataset twitch --epochs 100

python main.py --method gnnsafe --backbone gcn --device 0 --dataset twitch --epochs 100 \
  --use_energy --use_classifier

python main.py --method gnnsafe --backbone gcn --device 0 --dataset twitch --epochs 100 \
  --use_energy --use_classifier --use_energy_filter

python main.py --method gnnsafe --backbone gcn --device 0 --dataset twitch --epochs 100 \
  --use_energy --use_energy_propagation --use_classifier --use_energy_filter

python main.py --method gnnsafe --backbone gcn --device 0 --dataset twitch --epochs 100 \
  --use_energy --use_classifier --synthesis_ood

python main.py --method gnnsafe --backbone gcn --device 0 --dataset twitch --epochs 100 \
  --use_energy --use_classifier --use_energy_filter --synthesis_ood

python main.py --method gnnsafe --backbone gcn --device 0 --dataset twitch --epochs 100 \
  --use_energy --use_energy_propagation --use_classifier --use_energy_filter --synthesis_ood

echo -e "\nEnd training...\n"


python main.py --method "gnnsafe" --backbone "gcn" --dataset "cora" --ood_type "structure" --device 0 --epochs 100 --use_energy --use_energy_propagation --use_classifier --use_energy_filter --synthesis_ood --lower_bound_ood -9
