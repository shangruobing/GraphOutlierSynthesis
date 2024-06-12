# Graph Outlier Synthesis

***Out-of-Distribution Detection for Graph Neural Networks***

# Methodology

**通过KNN生成OOD数据集，然后使用GNN进行嵌入，嵌入结果使用`Energy`进行过滤，最后通过分类器进行分类。**

如果生成的OOD数据足够准确，这种数据增强的方法可以提升原方法的准确度，最差情况下也能持平。
但如果OOD数据存在错误，会导致模型误分类，从而降低原有方法的准确度。
为了解决这个问题，可以在生成数据之后增加一个数据质量判断方法，使用能量函数对生成的合成数据进行筛选。
筛选条件是能量函数转换后的值大于或小于某个阈值，只有满足条件的数据才可以用于模型训练。

## Steps

- **Identify graph data samples near the boundary**
- **Synthesize outliers based on boundary samples**
- **Filter the outliers by energy function**
- **Training Graph nerual network with Synthesize outliers**


# Environment

## Dev

- Ubuntu 20.04.3 LTS (GNU/Linux 5.15.0-91-generic x86_64)
- NVIDIA GeForce RTX 2060 6G
- CUDA NVIDIA-SMI 525.116.03 Driver Version: 525.116.03 CUDA Version: 12.0
- Python 3.10.9 (main, Mar 1 2023, 18:23:06) [GCC 11.2.0] on linux

# Prod

- CentOS Linux 7 (GNU/Linux 3.10.0-1160.el7.x86_64)
- Tesla V100-SXM2-32GB
- CUDA NVIDIA-SMI 460.106.00 Driver Version: 460.106.00 CUDA Version: 11.2
- Python 3.9.2 (default, Mar 3 2021, 20:02:32) [GCC 7.3.0] :: Anaconda, Inc. on linux

# Dependency

- torch==2.2.1
- torch_geometric==2.5.0
- torch_sparse==0.6.18+pt22cu121
- torch_scatter==2.1.2+pt22cu121

> https://data.pyg.org/whl/

# Install

We recommend to use `conda`.

```shell
conda env create -f environment.yml
```

Use `pip` may cause some problems when installing `torch_sparse` and `torch_scatter`.

```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch_geometric==2.5.0
pip install torch_sparse==0.6.18
pip install torch_scatter==2.1.0
```

# Run

```shell
# if you use conda
conda activate GraphOutlierSynthesis
# if you use venv
source venv/bin/activate

python main.py --method "gnnsafe" --backbone "gcn" --dataset "cora" --ood_type "knn" --device 0 --epochs 100 --use_classifier

# if you want to run by script
cd GraphOOD-GNNSafe/script
bash detect.sh

# if you want to run in background
nohup bash detect.sh >output.log 2>&1 &

```

# Common Command

```shell
nvidia-smi
nvidia-smi --query-gpu=name --format=csv,noheader
watch -n 2 -d nvidia-smi
```

# Dataset

| name         | num_nodes | num_features | num_classes | num_edges |
|--------------|-----------|--------------|-------------|-----------|
| Cora         | 1433      | 2708         | 7           | 10556     |
| Amazon-Photo | 7650      | 745          | 8           | 238162    |
| Actor        | 7600      | 932          | 5           | 30019     |
| GitHub       | 37000     | 128          | 2           | 578006    |      
| Wiki-CS      | 11701     | 300          | 10          | 431726    |   
| Coauthor-CS  | 18333     | 6805         | 15          | 163788    |

> wiki-cs和actor的mask不一致
