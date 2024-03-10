# conda deactivate && cd RuobingShang && source venv/bin/activate && cd GraphOOD-GNNSafe/GNNSafe

# conda deactivate && cd RuobingShang && source venv/bin/activate
# cd GraphOOD-GNNSafe/script
# bash detect.sh

cd ../GNNSafe || exit

epochs=30
device=0
datasets=("cora" "actor")
ood_types=("structure" "feature" "label")

for dataset in "${datasets[@]}"; do
  for ood_type in "${ood_types[@]}"; do
    for method in "msp" "gnnsafe"; do
      for backbone in "gcn" "mlp" "gat" "mixhop" "gcnjk" "gatjk"; do
        python main.py \
          --method $method \
          --backbone $backbone \
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
    for method in "msp" "gnnsafe"; do
      for backbone in "gcn" "mlp" "gat" "mixhop" "gcnjk" "gatjk"; do
        python main.py \
          --method $method \
          --backbone $backbone \
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

for dataset in "${datasets[@]}"; do
  for ood_type in "${ood_types[@]}"; do
    for method in "OE" "ODIN" "Mahalanobis"; do
      for backbone in "gcn" "mlp" "gat"; do
        python main.py \
          --method $method \
          --backbone $backbone \
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
    for method in "OE" "ODIN" "Mahalanobis"; do
      for backbone in "gcn" "mlp" "gat"; do
        python main.py \
          --method $method \
          --backbone $backbone \
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
