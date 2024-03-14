from typing import Tuple
import faiss
import numpy as np
import torch
from faiss import GpuIndexFlat
from torch import Tensor
from torch.distributions import MultivariateNormal


def generate_outliers(
        dataset: Tensor,
        num_nodes: int,
        num_features: int,
        num_edges: int,
        k=300,
        top=200,
        cov_mat=0.1,
        sampling_ratio=1.0,
        pic_nums=10,
        device=torch.device("cpu"),
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Generate outliers using the KNN algorithm.
    Args:
        dataset: the input dataset
        num_nodes: the number of nodes
        num_features: the number of features
        num_edges: the number of edges
        k: The number of nearest neighbors to return
        top: How many ID samples to pick to define as points near the boundary of the sample space
        cov_mat: The weight before the covariance matrix to determine the sampling range
        sampling_ratio: Sampling ratio
        pic_nums: Number of ID samples used to generate outliers
        device: torch device
    Returns:
        the generated outliers, the edges of the generated outliers, the labels of the generated outliers
    """

    # 将输入的数据集归一化，根据采样率选择数据集范围，然后存入向量库
    resource = faiss.StandardGpuResources()
    faiss_index = faiss.GpuIndexFlatL2(resource, num_features)
    normed_data = dataset / torch.norm(dataset, p=2, dim=1, keepdim=True)
    # 随机生成所要采样的数据集索引列表
    rand_index = np.random.choice(num_nodes, int(num_nodes * sampling_ratio), replace=False)
    faiss_index.add(normed_data[rand_index].cpu().numpy())

    # 找出数据集内每个元素最边界的k个元素，再从中挑取top个，做到挑选整个数据集最边界的元素
    max_distance_index, max_distance = search_max_distance(
        target=dataset,
        index=faiss_index,
        k=k,
        top=top
    )

    # 到此找到了最远的数据点，从最远的集合中随机选择10个
    max_distance_index = max_distance_index[np.random.choice(top, pic_nums, replace=False)]

    # 生成一个多元高斯分布，均值为0，协方差矩阵为单位矩阵
    gaussian_distribution = MultivariateNormal(
        loc=torch.zeros(num_features, device=device),
        covariance_matrix=torch.eye(num_features, device=device)
    )
    # 从高斯分布采样1/2节点数目的数据作为噪声2708 // 2 = 1354
    noises = gaussian_distribution.rsample(sample_shape=(num_nodes // 2,))

    # 噪声个数 1354
    num_noises = noises.shape[0]

    # 重复10次，每一次都补足为采样的length，把这10次结果拼起来
    # 1433个特征，复制1354行，10个就是13540行，共10组特征
    sample_points = torch.cat([
        dataset[index].repeat(num_noises, 1) for index in max_distance_index
    ])

    # 负采样同样重复10次，变成10组，13540行
    noise_cov = cov_mat * noises.repeat(pic_nums, 1)
    # 采样点加入噪声
    sample_points += noise_cov

    nums_point = num_nodes // 20
    sample_points = generate_negative_samples(
        target=sample_points,
        index=faiss_index,
        k=k,
        num_points=nums_point,
        num_negative_samples=num_noises
    )

    faiss_index.reset()

    num_sample_points = sample_points.shape[0]
    edge_node_radio = num_edges // num_nodes
    sample_edges = torch.randint(
        low=0,
        high=num_sample_points - 1,
        size=(2, edge_node_radio * num_sample_points),
        device=device
    )
    sample_labels = torch.zeros(size=num_sample_points, dtype=torch.long, device=device)
    return sample_points, sample_edges, sample_labels


def search_max_distance(
        target: Tensor,
        index: GpuIndexFlat,
        k: int,
        top: int
) -> Tuple[Tensor, Tensor]:
    """
    从数据集中找到每个元素的k个邻居，再选择每个元素距离最远的（第k个）邻居，最终从邻居中选择最远的num_selects个邻居
                        [[B,A],           [[A],
    [[A,B,C,D,E]]   =>   [C,D]       =>    [D],     => [B,D]
                         [E,B]]            [B]]
    Args:
        target: the target of the search
        index: The index structure used for the search.
        k: The number of nearest neighbors to return.
        top: The number of points to return from the nearest neighbors.

    Returns: max_distance_index, max_distance_value

    """
    normed_target = target / torch.norm(target, p=2, dim=1, keepdim=True)
    # 找出每个元素最近的k个元素，返回距离列表和索引列表
    distance, output_index = index.search(normed_target.cpu().numpy(), k)
    # 取每个元素最后（最远）的一个元素
    k_th_distance = torch.tensor(distance[:, -1])
    # 找出整个数据集最远的select个元素
    max_distance, max_distance_index = torch.topk(k_th_distance, top)
    return max_distance_index, max_distance


def generate_negative_samples(
        target: Tensor,
        index: GpuIndexFlat,
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

    """
    负样本数目为nums_negative_samples，13540
    nums_point为每个点，2706//20=1354
    故分为13540/1354=10组
    """
    normed_target = target / torch.norm(target, p=2, dim=1, keepdim=True)
    distance, output_index = index.search(normed_target.cpu().numpy(), k)
    # 取每个元素最后（最远）的一个元素，取k个邻居的最后一个
    k_th_distance = torch.tensor(distance[:, -1])
    # 特征为横向排列，如下变化
    """
    k_th_distance torch.Size([13540]) 
    k_th torch.Size([1354, 10])
    nums => nums_negative_samples,pic_nums
    [A,B,C,D,E,F]    =>    [[A,B],[C,D],[E,F]]
    """
    k_th = k_th_distance.view(num_negative_samples, -1)

    # 挑选负样本个数
    distance, minD_idx = torch.topk(k_th, num_points, dim=0)
    point_list = [i * num_negative_samples + minD_idx[:, i] for i in range(minD_idx.shape[1])]
    return target[torch.cat(point_list)]
