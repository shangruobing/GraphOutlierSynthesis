import numpy as np
import torch


def KNN_dis_search_decrease(target, index, K=50, select=1, ):
    """
    data_point: Queue for searching k-th points
    target: the target of the search
    K
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
    # print("distance", distance)
    # print("output_index", output_index)
    k_th_distance = distance[:, -1]
    k_th_distance, minD_idx = torch.topk(torch.Tensor(k_th_distance), select)
    # print("minD_idx", minD_idx)
    # print("k_th_distance", k_th_distance)
    return minD_idx, k_th_distance


def KNN_dis_search_distance(target, index, K=50, num_points=10, length=2000, depth=342):
    """
    data_point: Queue for searching k-th points
    target: the target of the search
    K
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


def generate_outliers(ID, input_index, negative_samples, ID_points_num=2, K=20, select=1, cov_mat=0.1,
                      sampling_ratio=1.0, pic_nums=30, depth=342):
    """
    ID: the input data 分布内数据 torch.Size([1000, 512])
    input_index: Faiss Index
    negative_samples: 进行随机采样得到负样本 torch.Size([600, 512])
    ID_points_num: the number of synthetic outliers extracted for each selected ID 从选择的分布内数据抽取的边界样本 2
    K: KNN距离 300
    select: How many ID samples to pick to define as points near the boundary of the sample space 多少ID样本用来定义边界 200
    cov_mat: The weight before the covariance matrix to determine the sampling range 采样范围0.1
    sampling_ratio: 采样率 1
    pic_nums: Number of ID samples used to generate outliers 2
    depth: 特征维度 932
    """
    # length 600个样本
    length = negative_samples.shape[0]
    # 归一化 torch.Size([7600, 932])
    normed_data = ID / torch.norm(ID, p=2, dim=1, keepdim=True)
    # 7600, 7600*1
    rand_ind = np.random.choice(normed_data.shape[0], int(normed_data.shape[0] * sampling_ratio), replace=False)
    # (7600,) print(rand_ind.shape)
    index = input_index
    # target.shape torch.Size([7600, 932])
    index.add(normed_data[rand_ind].cpu().numpy())

    # minD_idx,k_th 200,200
    minD_idx, k_th = KNN_dis_search_decrease(
        # 分布内数据 torch.Size([7600, 932])
        ID,
        # faiss
        index,
        # 300的距离
        K,
        # 选择200个
        select
    )

    minD_idx = minD_idx[np.random.choice(select, int(pic_nums), replace=False)]
    data_point_list = torch.cat([ID[i:i + 1].repeat(length, 1) for i in minD_idx])
    # data_point_list torch.Size([1200, 932])
    negative_sample_cov = cov_mat * negative_samples.to("cuda:0").repeat(pic_nums, 1)
    # negative_sample_cov torch.Size([1200, 932])
    negative_sample_list = negative_sample_cov + data_point_list
    # negative_sample_list torch.Size([1200, 932])
    point = KNN_dis_search_distance(negative_sample_list, index, K, ID_points_num, length, depth)
    index.reset()
    # point torch.Size([4, 932])
    return point


def generate_outliers_OOD(ID, input_index, negative_samples, K=100, select=100, sampling_ratio=1.0):
    data_norm = torch.norm(ID, p=2, dim=1, keepdim=True)
    normed_data = ID / data_norm
    rand_ind = np.random.choice(normed_data.shape[1], int(normed_data.shape[1] * sampling_ratio), replace=False)
    index = input_index
    index.add(normed_data[rand_ind])
    minD_idx, k_th = KNN_dis_search_decrease(negative_samples, index, K, select)

    return negative_samples[minD_idx]


def generate_outliers_rand(ID, input_index,
                           negative_samples, ID_points_num=2, K=20, select=1,
                           cov_mat=0.1, sampling_ratio=1.0, pic_nums=10,
                           repeat_times=30, depth=342):
    length = negative_samples.shape[0]
    data_norm = torch.norm(ID, p=2, dim=1, keepdim=True)
    normed_data = ID / data_norm
    rand_ind = np.random.choice(normed_data.shape[1], int(normed_data.shape[1] * sampling_ratio), replace=False)
    index = input_index
    index.add(normed_data[rand_ind])
    minD_idx, k_th = KNN_dis_search_decrease(ID, index, K, select)
    ID_boundary = ID[minD_idx]
    negative_sample_list = []
    for i in range(repeat_times):
        select_idx = np.random.choice(select, int(pic_nums), replace=False)
        sample_list = ID_boundary[select_idx]
        mean = sample_list.mean(0)
        var = torch.cov(sample_list.T)
        var = torch.mm(negative_samples, var)
        trans_samples = mean + var
        negative_sample_list.append(trans_samples)
    negative_sample_list = torch.cat(negative_sample_list, dim=0)
    point = KNN_dis_search_distance(negative_sample_list, index, K, ID_points_num, length, depth)

    index.reset()

    # return ID[minD_idx]
    return point
