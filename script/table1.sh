cd ../GNNSafe || exit

epochs=100
device=3
datasets=("cora" "coauthor-cs" "actor" "amazon-photo" "github" "wiki-cs")
ood_types=("structure")
backbones=("gcn")
methods=("msp" "gnnsafe" "OE" "ODIN" "Mahalanobis" "maxlogits" "energymodel" "energyprop")

for dataset in "${datasets[@]}"; do
  for ood_type in "${ood_types[@]}"; do
    for method in "${methods[@]}"; do
      for backbone in "${backbones[@]}"; do
        python main.py \
          --method "$method" \
          --backbone "$backbone" \
          --dataset "$dataset" \
          --ood_type "$ood_type" \
          --mode detect \
          --use_bn \
          --device $device \
          --epochs $epochs \
          --generate_ood
      done
    done
  done
done

for dataset in "${datasets[@]}"; do
  for ood_type in "${ood_types[@]}"; do
    for method in "${methods[@]}"; do
      for backbone in "${backbones[@]}"; do
        python main.py \
          --method "$method" \
          --backbone "$backbone" \
          --dataset "$dataset" \
          --ood_type "$ood_type" \
          --mode detect \
          --use_bn \
          --device $device \
          --epochs $epochs
      done
    done
  done
done
