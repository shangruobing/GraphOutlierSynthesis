conda deactivate && cd RuobingShang && source venv/bin/activate && cd GraphOOD-GNNSafe/GNNSafe

for data in "cora"; do
    for ood in "structure"; do
        for backbone in "gcn" "mlp" "gat" "mixhop" "gcnjk" "gatjk"; do
            python main.py \
                --method gnnsafe \
                --backbone $backbone \
                --dataset $data \
                --ood_type $ood \
                --mode detect \
                --use_bn \
                --device 0 \
                --epochs 100 \
                --generate_ood
        done
    done
done


for data in "cora"; do
    for ood in "structure"; do
        for backbone in "gcn" "mlp" "gat" "mixhop" "gcnjk" "gatjk"; do
            python main.py \
                --method gnnsafe \
                --backbone $backbone \
                --dataset $data \
                --ood_type $ood \
                --mode detect \
                --use_bn \
                --device 0 \
                --epochs 100
        done
    done
done
