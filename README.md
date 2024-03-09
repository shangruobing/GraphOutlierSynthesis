# Graph Outlier Synthesis

***Out-of-Distribution Detection for Graph Neural Networks***

# Experiments
```shell
conda deactivate && cd RuobingShang && source venv/bin/activate && cd GraphOOD-GNNSafe/GNNSafe
```


```shell
python main.py \
  --method gnnsafe \
  --backbone gcn \
  --dataset cora \
  --ood_type feature \
  --mode detect \
  --use_bn \
  --device 0 \
  --epochs 100
  
  
python main.py \
  --method gnnsafe \
  --backbone gcn \
  --dataset cora \
  --ood_type feature \
  --mode detect \
  --use_bn \
  --epochs 500 \
  --cpu
  
python main.py \
  --method gnnsafe \
  --backbone gcn \
  --dataset cora \
  --ood_type feature \
  --mode detect \
  --use_bn \
  --epochs 500 \
  --cpu \
  --generate_ood false
```

| Epoch | Score | OOD |
|-------|-------|-----|
| 10    | 27.60 | F   |
| 12    | 25.15 | T   |
|       |       |     |
|       |       |     |
|       |       |     |
|       |       |     |
|       |       |     |
