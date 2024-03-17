# conda deactivate && cd RuobingShang && source venv/bin/activate && cd GraphOOD-GNNSafe/GNNSafe

# conda deactivate && cd RuobingShang && source venv/bin/activate
# cd GraphOOD-GNNSafe/script
# bash detect.sh
# python main.py \
# --method gnnsafe --backbone gcn --dataset cora --ood_type label --mode detect --use_bn --device 3 --epochs 50 --generate_ood

cd ../GNNSafe || exit

epochs=50
device=3
datasets=("coauthor-cs")
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
