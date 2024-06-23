from typing import Tuple

import numpy as np
import faiss
from faiss import IndexFlat
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal


def generate_outliers(
        dataset: Tensor,
        num_nodes: int,
        num_features: int,
        num_edges: int,
        # num_classes: int = 2,
        cov_mat=0.1,
        sampling_ratio=1.0,
        device=torch.device("cpu"),
        debug=False
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Generate outliers using the KNN algorithm.
    我们利用 ID 数据生成 OOD 数据。
    首先，对输入数据进行归一化处理，然后根据设定的采样率参数，选取部分数据存入 Faiss 中等待检索。
    我们使用 KNN 找到每个元素的 K 个最近邻，然后选择其最远的邻居。这一过程有助于找到处于数据集边界的样本。
    然后，我们使用均值为 0、协方差矩阵为单位矩阵的多元高斯分布生成噪声点。
    将找到的边界点和噪声点合并，形成采样点集合。
    我们再次利用 Faiss 进行检索，找到每个采样点的 K 个最近邻，然后选择其最远的邻居。这一过程有助于找到加入噪音点后的采样点。
    我们将利用这个采样点集合与原始数据集中边数目与节点数目的比例生成采样边。
    这个步骤生成的OOD数据集将被输入encoder，从而得到当前模型对于OOD数据预测的loss。
    Args:
        # num_classes:
        dataset: the input dataset
        num_nodes: the number of nodes
        num_features: the number of features
        num_edges: the number of edges
        # k: The number of nearest neighbors to return
        # top: How many ID samples to pick to define as points near the boundary of the sample space
        cov_mat: The weight before the covariance matrix to determine the sampling range
        sampling_ratio: Sampling ratio
        # pic_nums: Number of ID samples used to generate outliers
        device: torch device
        debug:
    Returns:
        the generated outliers, the edges of the generated outliers, the labels of the generated outliers
    """

    # How many ID samples to pick to define as points near the boundary of the sample space
    top = num_nodes // 10
    # The number of nearest neighbors to return
    k = min(len(dataset), 100)

    # Number of ID samples used to generate outliers
    pic_nums = num_nodes // 20

    num_sample_points = int(num_nodes * sampling_ratio)

    # 将输入的数据集归一化，根据采样率选择数据集范围，然后存入向量库
    faiss_index = faiss.IndexFlatL2(num_features)
    if device.type != "cpu":
        resource = faiss.StandardGpuResources()
        faiss_index = faiss.index_cpu_to_gpu(provider=resource, device=device.index, index=faiss_index)

    # 随机生成所要采样的数据集索引列表
    rand_index = np.random.choice(num_nodes, num_sample_points, replace=False)
    faiss_index.add(dataset[rand_index].cpu().numpy())

    # 找出数据集内每个元素最边界的k个元素，再从中挑取top个，做到挑选整个数据集最边界的元素
    max_distance_index, max_distance = search_max_distance(
        target=dataset,
        index=faiss_index,
        k=k,
        top=top
    )

    all_max_distance_index = max_distance_index

    # 到此找到了最远的数据点，从最远的集合中随机选择
    max_distance_index = max_distance_index[np.random.choice(top, pic_nums, replace=False)]

    # 重复10次，每一次都补足为采样的length，把这10次结果拼起来，每个采样点贡献10个特征
    sampling_dataset = torch.cat([
        dataset[index].repeat(num_nodes // pic_nums, 1) for index in max_distance_index
    ])

    # 生成一个多元高斯分布，均值为0，协方差矩阵为单位矩阵
    gaussian_distribution = MultivariateNormal(
        loc=torch.zeros(num_features, device=device),
        covariance_matrix=torch.eye(num_features, device=device)
    )

    # 从高斯分布采样作为噪声
    noises = gaussian_distribution.rsample(sample_shape=(sampling_dataset.shape[0],))

    noise_cov = cov_mat * noises
    # 采样点加入噪声
    sampling_dataset += noise_cov

    sample_points = generate_negative_samples(
        target=sampling_dataset,
        index=faiss_index,
        k=k,
        num_points=pic_nums,
        num_negative_samples=num_sample_points
    )

    faiss_index.reset()

    num_sample_points = sample_points.shape[0]

    # 通过计算分布内数据中的边点比(边的数目//节点的数目)来生成对应数目的边
    edge_node_radio = num_edges // num_nodes
    sample_edges = torch.randint(
        low=0,
        high=num_sample_points - 1,
        size=(2, int(edge_node_radio * num_sample_points)),
        device=device
    )

    # sample_labels = torch.randint(
    #     low=0,
    #     high=num_classes,
    #     size=(int(num_sample_points),),
    #     device=device
    # )

    # 由于不将其带入监督训练过程，假设Label全为0，生成的都是0的label
    sample_labels = torch.zeros(num_sample_points, dtype=torch.long, device=device)

    if debug:
        return sample_points, sample_edges, sample_labels, all_max_distance_index, max_distance_index, f"top:{top} k:{k} pic_nums:{pic_nums} dataset:{dataset.size()} sample_points:{sample_points.size()}"
    else:
        return sample_points, sample_edges, sample_labels


def search_max_distance(
        target: Tensor,
        index: IndexFlat,
        k: int,
        top: int
) -> Tuple[Tensor, Tensor]:
    """
    从数据集中找到每个元素的k个邻居，再选择每个元素距离最远的（第k个）邻居，最终从邻居中选择最远的num_selects个邻居
    k = 2
                        [[B,C],           [[C],
    [[A,B,C]]   =>       [A,C]       =>    [C],     => [B,C]
                         [A,B]]            [B]]
    Args:
        target: the target of the search
        index: The index structure used for the search.
        k: The number of nearest neighbors to return.
        top: The number of points to return from the nearest neighbors.

    Returns: max_distance_index, max_distance_value

    """
    # 找出每个元素最近的k个元素，返回距离列表和索引列表
    distance, output_index = index.search(target.cpu().numpy(), k)
    # 取每个元素最后（最远）的一个元素
    k_th_distance = torch.tensor(distance[:, -1])
    # ic(k_th_distance)
    # 找出整个数据集最远的select个元素
    max_distance, max_distance_index = torch.topk(k_th_distance, top)
    # ic(max_distance_index)
    return max_distance_index, max_distance


def generate_negative_samples(
        target: Tensor,
        index: IndexFlat,
        k: int,
        num_points: int,
        num_negative_samples: int
) -> Tensor:
    """

    Args:
        target: the target of the search. ([13540, 1433])
        index: The index structure used for the search.
        k: 300
        num_points: The number of points to return from the nearest neighbors. 135
        num_negative_samples: 1354

    Returns: sample_points

    """
    distance, output_index = index.search(target.cpu().numpy(), k)
    # 取每个元素最后（最远）的一个元素，取k个邻居的最后一个
    k_th_distance = torch.tensor(distance[:, -1])
    # 特征为横向排列，如下变化
    """
    k_th_distance torch.Size([13540]) 
    k_th torch.Size([1354, 10])
    nums => nums_negative_samples,pic_nums
    [A,B,C,D,E,F]    =>    [[A,B],[C,D],[E,F]]
    """
    # max_distance, max_distance_index = torch.topk(k_th_distance, k_th_distance.shape[0], dim=0)
    max_distance, max_distance_index = torch.topk(k_th_distance, num_points, dim=0)

    outliers = target[max_distance_index].repeat(repeats=(num_negative_samples // num_points, 1))
    nosies = torch.cat([torch.rand_like(outliers) * -0.05, torch.rand_like(outliers) * 0.05])
    shape = nosies.shape
    flattened_tensor = nosies.view(-1)
    # 随机打乱噪声的顺序
    shuffled_indices = torch.randperm(flattened_tensor.size(0))
    shuffled_tensor = flattened_tensor[shuffled_indices]
    nosies = shuffled_tensor.view(shape)[:shape[0] // 2, :]
    return outliers + nosies


# def normalize(dataset: Tensor) -> Tensor:
#     min_value = dataset.min(dim=0)[0]
#     max_value = dataset.max(dim=0)[0]
#     normalized_dataset = (dataset - min_value) / (max_value - min_value)
#     normalized_dataset[torch.isnan(normalized_dataset)] = 0
#     return normalized_dataset


if __name__ == '__main__':
    from src.common.visualize import visualize_2D

    dataset = torch.rand(2000, 2)

    num_nodes, num_features = dataset.shape[0], dataset.shape[1]
    num_edges = 10
    sample_point, sample_edge, sample_label, all_max_distance_index, max_distance_index, title = generate_outliers(
        dataset=dataset,
        num_nodes=num_nodes,
        num_features=num_features,
        num_edges=num_edges,
        debug=True,
    )

    visualize_2D(dataset=dataset.cpu(), all_boundary=all_max_distance_index, boundary=max_distance_index, outlier=sample_point.cpu(), title=title)

    # dataset = torch.rand(500, 3)
    # num_nodes, num_features = dataset.shape[0], dataset.shape[1]
    # num_edges = 10
    # sample_point, sample_edge, sample_label, all_max_distance_index, max_distance_index, title = generate_outliers(
    #     dataset=dataset,
    #     num_nodes=num_nodes,
    #     num_features=num_features,
    #     num_edges=num_edges,
    #     debug=True,
    # )
    #
    # visualize_3D(dataset=dataset.cpu(), all_boundary=all_max_distance_index, boundary=max_distance_index, outlier=sample_point.cpu(), title=title)
