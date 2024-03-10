# Graph Outlier Synthesis

***Out-of-Distribution Detection for Graph Neural Networks***

# Environment

- Ubuntu 20.04.3 LTS (GNU/Linux 5.15.0-91-generic x86_64)
- NVIDIA GeForce RTX 2060 6G
- CUDA NVIDIA-SMI 525.116.03 Driver Version: 525.116.03 CUDA Version: 12.0
- Python 3.10.9 (main, Mar  1 2023, 18:23:06) [GCC 11.2.0] on linux

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