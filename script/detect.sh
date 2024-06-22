cd ../GNNSafe || exit

epochs=100
device=1
#datasets=("cora" "amazon-photo" "github" "coauthor-cs")
datasets=("cora")
ood_types=("knn")
#methods=("msp" "OE" "ODIN" "Mahalanobis" "maxlogits" "energymodel" "energyprop" "gnnsafe")
methods=("msp")
#backbones=("gcn")
backbones=("mlp" "sgc" "gcn" "gat" "mixhop" "gcnjk" "gatjk" "H2GCNConv" "APPNP_Net" "GPRPROP" "GPRGNN")

echo "Parameters:"
echo epochs: $epochs
echo device: $device
echo datasets: "${datasets[@]}"
echo ood_types: "${ood_types[@]}"
echo methods: "${methods[@]}"
echo backbones: "${backbones[@]}"
echo -e "\nStart training...\n"

python main.py --method "msp" --backbone "gcn" --dataset "cora" --ood_type "structure" --device 0 --epochs 10

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
          --use_classifier

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