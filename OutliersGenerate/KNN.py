import faiss
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch_geometric.data import Dataset


def KNN_dis_search_decrease(target, index, K=50, select=1, ):
    """

    Args:
        target: the target of the search
        index:
        K:
        select:

    Returns:

    """
    # Normalize the features
    # print("target.shape", target.shape)
    # print("target", target)
    # print("index", index)
    target_norm = torch.norm(target, p=2, dim=1, keepdim=True)
    normed_target = target / target_norm
    # print("normed_target", normed_target)
    # print("normed_target.shape", normed_target.shape)
    # print("K", K)
    distance, output_index = index.search(normed_target.cpu().numpy(), K)
    # distance, output_index = index.search(normed_target, K)
    # print("distance", distance)
    # print("output_index", output_index)
    k_th_distance = distance[:, -1]
    k_th_distance, minD_idx = torch.topk(torch.Tensor(k_th_distance), select)
    # print("minD_idx", minD_idx)
    # print("k_th_distance", k_th_distance)
    return minD_idx, k_th_distance


def KNN_dis_search_distance(target, index, K=50, num_points=10, length=2000, depth=342):
    """

    Args:
        target: the target of the search
        index:
        K:
        num_points:
        length:
        depth:

    Returns:

    """
    # Normalize the features
    #
    # print("target", target)
    # print("index", index)

    normed_target = target / torch.norm(target, p=2, dim=1, keepdim=True)
    distance, output_index = index.search(normed_target.cpu().numpy(), K)
    # distance, output_index = index.search(normed_target, K)
    k_th_distance = torch.Tensor(distance[:, -1])
    k_th = k_th_distance.view(length, -1)
    k_th_distance, minD_idx = torch.topk(k_th, num_points, dim=0)
    minD_idx = minD_idx.squeeze()
    point_list = []
    for i in range(minD_idx.shape[1]):
        point_list.append(i * length + minD_idx[:, i])
    return target[torch.cat(point_list)]


def generate_outliers(
        dataset: torch.Tensor,
        num_nodes,
        num_features,
        num_edges,
        distance=300,
        select=200,
        cov_mat=0.1,
        sampling_ratio=1.0,
        pic_nums=10,
        device=torch.device("cpu"),
):
    """
    Args:
        dataset: the input dataset
        num_nodes: the number of nodes
        num_features: the number of features
        num_edges: the number of edges
        distance: KNN distance
        select: How many ID samples to pick to define as points near the boundary of the sample space
        cov_mat: The weight before the covariance matrix to determine the sampling range
        sampling_ratio: Sampling ratio
        pic_nums: Number of ID samples used to generate outliers
        device: torch device
    Returns:

    """
    # feature dimension
    # num_features = dataset.shape[1]
    # the number of synthetic outliers extracted for each selected ID
    outliers_per_ID = dataset.shape[0] // 20

    gaussian_distribution = MultivariateNormal(torch.zeros(num_features, device=device),
                                               torch.eye(num_features, device=device))
    # 采样1/2的数据
    negative_samples = gaussian_distribution.rsample((num_nodes // 2,))

    resource = faiss.StandardGpuResources()
    faiss_index = faiss.GpuIndexFlatL2(resource, num_features)

    length = negative_samples.shape[0]
    # 归一化 torch.Size([7600, 932])
    normed_data = dataset / torch.norm(dataset, p=2, dim=1, keepdim=True)
    # 7600, 7600*1
    rand_ind = np.random.choice(normed_data.shape[0], int(normed_data.shape[0] * sampling_ratio), replace=False)
    # (7600,) print(rand_ind.shape)
    # index = input_index
    # target.shape torch.Size([7600, 932])
    faiss_index.add(normed_data[rand_ind].cpu().numpy())

    # minD_idx,k_th 200,200
    minD_idx, k_th = KNN_dis_search_decrease(
        # 分布内数据 torch.Size([7600, 932])
        dataset,
        # faiss
        faiss_index,
        # 300的距离
        distance,
        # 选择200个
        select
    )

    minD_idx = minD_idx[np.random.choice(select, int(pic_nums), replace=False)]
    data_point_list = torch.cat([dataset[i:i + 1].repeat(length, 1) for i in minD_idx])
    # data_point_list torch.Size([1200, 932])
    negative_sample_cov = cov_mat * negative_samples.to(device).repeat(pic_nums, 1)
    # negative_sample_cov torch.Size([1200, 932])
    negative_sample_list = negative_sample_cov + data_point_list
    # negative_sample_list torch.Size([1200, 932])
    sample_point = KNN_dis_search_distance(negative_sample_list, faiss_index, distance, outliers_per_ID, length,
                                           num_features)
    faiss_index.reset()

    sample_label = torch.zeros(sample_point.shape[0], dtype=torch.long, device=device).view(-1)
    edge_node_radio = num_edges // num_nodes
    sample_edge = torch.randint(
        low=0,
        high=len(sample_label) - 1,
        size=(2, edge_node_radio * len(sample_label)),
        device=device
    )
    # print("sample_point, sample_edge, sample_label", sample_point.size(), sample_edge.size(), sample_label.size())
    return sample_point, sample_edge, sample_label

# def generate_outliers_OOD(ID, input_index, negative_samples, K=100, select=100, sampling_ratio=1.0):
#     data_norm = torch.norm(ID, p=2, dim=1, keepdim=True)
#     normed_data = ID / data_norm
#     rand_ind = np.random.choice(normed_data.shape[1], int(normed_data.shape[1] * sampling_ratio), replace=False)
#     index = input_index
#     index.add(normed_data[rand_ind])
#     minD_idx, k_th = KNN_dis_search_decrease(negative_samples, index, K, select)
#
#     return negative_samples[minD_idx]
#
#
# def generate_outliers_rand(ID, input_index,
#                            negative_samples, ID_points_num=2, K=20, select=1,
#                            cov_mat=0.1, sampling_ratio=1.0, pic_nums=10,
#                            repeat_times=30, depth=342):
#     length = negative_samples.shape[0]
#     data_norm = torch.norm(ID, p=2, dim=1, keepdim=True)
#     normed_data = ID / data_norm
#     rand_ind = np.random.choice(normed_data.shape[1], int(normed_data.shape[1] * sampling_ratio), replace=False)
#     index = input_index
#     index.add(normed_data[rand_ind])
#     minD_idx, k_th = KNN_dis_search_decrease(ID, index, K, select)
#     ID_boundary = ID[minD_idx]
#     negative_sample_list = []
#     for i in range(repeat_times):
#         select_idx = np.random.choice(select, int(pic_nums), replace=False)
#         sample_list = ID_boundary[select_idx]
#         mean = sample_list.mean(0)
#         var = torch.cov(sample_list.T)
#         var = torch.mm(negative_samples, var)
#         trans_samples = mean + var
#         negative_sample_list.append(trans_samples)
#     negative_sample_list = torch.cat(negative_sample_list, dim=0)
#     point = KNN_dis_search_distance(negative_sample_list, index, K, ID_points_num, length, depth)
#
#     index.reset()
#
#     # return ID[minD_idx]
#     return point
