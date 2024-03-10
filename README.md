# Graph Outlier Synthesis

***Out-of-Distribution Detection for Graph Neural Networks***

# Environment

- Ubuntu 20.04.3 LTS (GNU/Linux 5.15.0-91-generic x86_64)
- NVIDIA GeForce RTX 2060 6G
- CUDA NVIDIA-SMI 525.116.03 Driver Version: 525.116.03 CUDA Version: 12.0
- Python 3.10.9 (main, Mar 1 2023, 18:23:06) [GCC 11.2.0] on linux

# Dependency

- torch==2.2.1
- torch_geometric==2.5.0
- torch_sparse==0.6.18+pt22cu121
- torch_scatter==2.1.2+pt22cu121

> https://data.pyg.org/whl/

# Install

```shell
python -m venv venv
pip install -r requirements.txt
```

# Run

```shell
conda deactivate && cd RuobingShang && source venv/bin/activate
cd GraphOOD-GNNSafe/script
bash detect.sh
```

# Common Command

```shell
nvidia-smi
nvidia-smi --query-gpu=name --format=csv,noheader
```

# Dataset
- Cora
- Actor