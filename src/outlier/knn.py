import logging
import time
from dataclasses import dataclass

import faiss
import numpy as np
import torch
from faiss import IndexFlat
from torch import Tensor
from torch.distributions import MultivariateNormal

__all__ = ["Outlier", "generate_outliers"]


@dataclass
class Outlier:
    sample_points: Tensor
    sample_edges: Tensor
    sample_labels: Tensor
    all_boundary_point_indices: Tensor
    selected_boundary_point_indices: Tensor


def generate_outliers(
        dataset: Tensor,
        num_nodes: int,
        num_features: int,
        num_edges: int,
        cov_mat=0.1,
        sampling_ratio=1.0,
        boundary_ratio=0.1,
        boundary_sampling_ratio=0.5,
        k=100,
        device=torch.device("cpu"),
) -> Outlier:
    """
    Generate outliers using the KNN algorithm.
    - We generate OOD data using ID data.
    - First, normalize the input data.
    - Select a subset of data based on the sampling rate and store it in Faiss for retrieval.
    - Use KNN to find the K nearest neighbors for each element, then select the farthest neighbor to identify boundary samples.
    - Generate noise points using a multivariate Gaussian distribution with a mean of 0 and a unit covariance matrix.
    - Combine the boundary points and noise points to form a sampling set.
    - Retrieve the K nearest neighbors for each sampling point using Faiss again, then select the farthest neighbor to refine the sampling set.
    - Generate sampling edges using the ratio of edges to nodes in the original dataset.
    Args:
        dataset: the input dataset
        num_nodes: the number of nodes
        num_features: the number of features
        num_edges: the number of edges
        cov_mat: The weight before the covariance matrix to determine the sampling range
        sampling_ratio: How many OOD samples to generate
        boundary_ratio: How many ID samples are defined as points near the boundary
        boundary_sampling_ratio: How many boundary used to generate outliers
        k: The number of nearest neighbors to return
        device: torch device
    Returns:
        the generated outliers, the edges of the generated outliers, the labels of the generated outliers
    """
    begin_time = time.time()
    dataset = dataset.clone().to(device)

    # How many OOD samples to generate
    num_sample_points = int(num_nodes * sampling_ratio)

    # How many ID samples are defined as points near the boundary
    num_boundary = int(num_nodes * boundary_ratio)

    # How many boundary used to generate outliers
    num_pick = int(num_boundary * boundary_sampling_ratio)

    # The number of nearest neighbors to return
    k = min(len(dataset), k)

    print("==================== Begin Generate Outliers ====================")
    print(f"Number of OOD samples to generate: {num_sample_points}")
    print(f"Number of ID samples defined as points near the boundary: {num_boundary}")
    print(f"Number of boundaries used to generate outliers: {num_pick}")
    print(f"Number of nearest neighbors to return: {k}")
    print(f"Device for generating outliers: {device.type}")

    faiss_index = faiss.IndexFlatL2(num_features)
    if "cuda" in device.type:
        resource = faiss.StandardGpuResources()
        if device.index:
            # Possible C/C++ prototypes are:
            # faiss::gpu::index_cpu_to_gpu(faiss::gpu::GpuResourcesProvider *,int,faiss::Index const *,faiss::gpu::GpuClonerOptions const *)
            # faiss::gpu::index_cpu_to_gpu(faiss::gpu::GpuResourcesProvider *,int,faiss::Index const *)
            faiss_index = faiss.index_cpu_to_gpu(provider=resource, device=device.index, index=faiss_index)
        else:
            faiss_index = faiss.index_cpu_to_gpu(provider=resource, device=0, index=faiss_index)

    # Randomly generate a list of indices of the dataset
    rand_index = np.random.choice(num_nodes, num_sample_points, replace=False)
    faiss_index.add(dataset[rand_index].cpu().numpy())

    # Find the k most marginal elements of each element in the data set, and then pick the top of them.
    # So that the most marginal elements of the entire data set are selected.
    all_boundary_point_indices, max_distance = search_max_distance(
        target=dataset,
        index=faiss_index,
        k=k,
        top=num_boundary
    )

    # The farthest data point is found here, and it is randomly selected from the farthest set.
    selected_boundary_point_indices = all_boundary_point_indices[np.random.choice(num_boundary, num_pick, replace=False)]

    # This is repeated n times, each time compensating for the length of the sample,
    # and the n results are pieced together, with each sample point contributing n features
    sampling_dataset = torch.cat([
        dataset[index].repeat(num_nodes // num_pick, 1) for index in selected_boundary_point_indices
    ])

    # A multivariate Gaussian distribution with mean 0 and covariance matrix as identity matrix is generated.
    try:
        gaussian_distribution = MultivariateNormal(
            loc=torch.zeros(num_features, device=device),
            covariance_matrix=torch.eye(num_features, device=device)
        )
    except NotImplementedError as e:
        logging.error(e, exc_info=True)
        candidate_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        gaussian_distribution = MultivariateNormal(
            loc=torch.zeros(num_features, device=candidate_device),
            covariance_matrix=torch.eye(num_features, device=candidate_device),
        )

    # Sampling from a Gaussian distribution as noise
    noises = gaussian_distribution.rsample(sample_shape=(sampling_dataset.shape[0],))

    noise_cov = cov_mat * noises
    # Noise is added to the sampling points
    sampling_dataset += noise_cov

    sample_points = generate_negative_samples(
        target=sampling_dataset,
        index=faiss_index,
        k=k,
        num_points=num_pick,
        num_negative_samples=num_sample_points
    )

    faiss_index.reset()

    num_sample_points = sample_points.shape[0]

    # The number of edges is generated by calculating the edge-to-point ratio (number of edges // number of nodes) of the data in the distribution
    edge_node_radio = num_edges // num_nodes
    sample_edges = torch.randint(
        low=0,
        high=num_sample_points - 1,
        size=(2, int(edge_node_radio * num_sample_points)),
        device=device
    )

    # Since it is not brought into the supervised training process, it is assumed that the labels are all 0, and the generated labels are all 0.
    sample_labels = torch.zeros(num_sample_points, dtype=torch.long, device=device)

    print(f"Time taken to generate outliers {round(time.time() - begin_time, 2)}s")
    print("===================== End Generate Outliers =====================")
    return Outlier(
        sample_points=sample_points,
        sample_edges=sample_edges,
        sample_labels=sample_labels,
        all_boundary_point_indices=all_boundary_point_indices,
        selected_boundary_point_indices=selected_boundary_point_indices
    )


def search_max_distance(
        target: Tensor,
        index: IndexFlat,
        k: int,
        top: int
) -> tuple[Tensor, Tensor]:
    """
    Find the k neighbors of each element in the dataset,
    then select the farthest (Kth) neighbor of each element,
    and finally select the farthest num_selects neighbors from the neighbors

    Such as:
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
    # Find the k nearest elements for each element and return a list of distances and indices
    target = target.cpu().numpy()
    if not target.flags.c_contiguous:
        target = target.copy(order='C')
    distance, output_index = index.search(target, k)
    # Take the last (farthest) element of each element
    k_th_distance = torch.tensor(distance[:, -1])
    # Find the farthest selected elements in the entire data set
    max_distance, max_distance_index = torch.topk(k_th_distance, top)
    return max_distance_index, max_distance


def generate_negative_samples(
        target: Tensor,
        index: IndexFlat,
        k: int,
        num_points: int,
        num_negative_samples: int
) -> Tensor:
    """
    Generate negative samples by adding noise to the target points.

    Args:
        target: the target of the search. ([13540, 1433])
        index: The index structure used for the search.
        k: 300
        num_points: The number of points to return from the nearest neighbors. 135
        num_negative_samples: 1354

    Returns: sample_points

    """
    distance, output_index = index.search(target.cpu().numpy(), k)
    # Take the last (farthest) element of each element and take the last of the k neighbors
    k_th_distance = torch.tensor(distance[:, -1])
    """
    The features are arranged horizontally as follows:
    k_th_distance torch.Size([13540]) 
    k_th torch.Size([1354, 10])
    nums => nums_negative_samples,pic_nums
    [A,B,C,D,E,F]    =>    [[A,B],[C,D],[E,F]]
    """
    max_distance, max_distance_index = torch.topk(k_th_distance, num_points, dim=0)

    outliers = target[max_distance_index].repeat(repeats=(num_negative_samples // num_points, 1))
    nosies = torch.cat([torch.rand_like(outliers) * -0.05, torch.rand_like(outliers) * 0.05])
    shape = nosies.shape
    flattened_tensor = nosies.view(-1)
    # Shuffle the order of the noise randomly
    shuffled_indices = torch.randperm(flattened_tensor.size(0))
    shuffled_tensor = flattened_tensor[shuffled_indices]
    nosies = shuffled_tensor.view(shape)[:shape[0] // 2, :]
    return outliers + nosies


if __name__ == '__main__':
    dataset = torch.rand(2000, 2)

    num_nodes, num_features = dataset.shape[0], dataset.shape[1]
    num_edges = 10
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    outliers = generate_outliers(
        dataset=dataset,
        num_nodes=num_nodes,
        num_features=num_features,
        num_edges=num_edges,
        device=device
    )
