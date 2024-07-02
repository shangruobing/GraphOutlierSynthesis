cd ../src || exit

epochs=100
device=0
#datasets=("cora" "amazon-photo" "github" "coauthor-cs")
datasets=("cora")
ood_types=("structure" "feature" "label")
#methods=("msp" "OE" "ODIN" "Mahalanobis" "maxlogits" "energymodel" "energyprop" "gnnsafe")
methods=("gnnsafe")
#backbones=("mlp" "sgc" "gcn" "gat" "mixhop" "gcnjk" "gatjk" "H2GCNConv" "APPNP_Net" "GPRPROP" "GPRGNN")
backbones=("gcn")

echo "Parameters:"
echo epochs: $epochs
echo device: $device
echo datasets: "${datasets[@]}"
echo ood_types: "${ood_types[@]}"
echo methods: "${methods[@]}"
echo backbones: "${backbones[@]}"
echo -e "\nStart training...\n"

for dataset in "${datasets[@]}"; do
  for ood_type in "${ood_types[@]}"; do
    for method in "${methods[@]}"; do
      for backbone in "${backbones[@]}"; do
        echo "Generate OOD data"
        date
        python main.py \
          --method "$method" \
          --backbone "$backbone" \
          --dataset "$dataset" \
          --ood_type "$ood_type" \
          --device $device \
          --epochs $epochs \
          --use_energy \
          --use_energy_propagation \
          --use_classifier \
          --use_energy_filter \
          --synthesis_ood

        echo "No Generate OOD data"
        date
        python main.py \
          --method "$method" \
          --backbone "$backbone" \
          --dataset "$dataset" \
          --ood_type "$ood_type" \
          --device $device \
          --epochs $epochs
      done
    done
  done
done

echo -e "\nEnd training...\n"
