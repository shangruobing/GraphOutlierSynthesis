# conda deactivate && cd RuobingShang && source venv/bin/activate && cd GraphOOD-GNNSafe/GNNSafe

# conda deactivate && cd RuobingShang && source venv/bin/activate
# cd GraphOOD-GNNSafe/script
# bash detect.sh

cd ../GNNSafe || exit

epochs=50
device=0

for data in "cora"; do
  for ood in "structure"; do
    for method in "msp" "gnnsafe"; do
      for backbone in "gcn" "mlp" "gat" "mixhop" "gcnjk" "gatjk"; do
        python main.py \
          --method $method \
          --backbone $backbone \
          --dataset $data \
          --ood_type $ood \
          --mode detect \
          --use_bn \
          --device $device \
          --epochs $epochs \
          --generate_ood
      done
    done
  done
done

for data in "cora"; do
  for ood in "structure"; do
    for method in "msp" "gnnsafe"; do
      for backbone in "gcn" "mlp" "gat" "mixhop" "gcnjk" "gatjk"; do
        python main.py \
          --method $method \
          --backbone $backbone \
          --dataset $data \
          --ood_type $ood \
          --mode detect \
          --use_bn \
          --device $device \
          --epochs $epochs
      done
    done
  done
done

for data in "cora"; do
  for ood in "structure"; do
    for method in "OE" "ODIN" "Mahalanobis"; do
      for backbone in "gcn" "mlp" "gat"; do
        python main.py \
          --method $method \
          --backbone $backbone \
          --dataset $data \
          --ood_type $ood \
          --mode detect \
          --use_bn \
          --device $device \
          --epochs $epochs \
          --generate_ood
      done
    done
  done
done

for data in "cora"; do
  for ood in "structure"; do
    for method in "OE" "ODIN" "Mahalanobis"; do
      for backbone in "gcn" "mlp" "gat"; do
        python main.py \
          --method $method \
          --backbone $backbone \
          --dataset $data \
          --ood_type $ood \
          --mode detect \
          --use_bn \
          --device $device \
          --epochs $epochs
      done
    done
  done
done
