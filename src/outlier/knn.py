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
        num_classes: int = 1,
        cov_mat=0.1,
        sampling_ratio=1.0,
        boundary_ratio=0.1,
        boundary_sampling_ratio=0.5,
        k=100,
        device=torch.device("cpu"),
) -> Outlier:
    """
    Generate outliers using the KNN algorithm.
    - Select a subset of data based on the sampling rate and store it in Faiss for retrieval.
    - Use KNN to find the K nearest neighbors for each element, and select the farthest neighbor as boundary samples.
    - Generate noise points from a multivariate Gaussian distribution with a mean of 0 and an identity covariance matrix.
    - Merge the boundary points and noise points to create a sampling set.
    - Retrieve the K nearest neighbors for each sampling point, and select the farthest neighbor to refine the sampling set.
    - Generate sampling edges and labels based on the ratio of edges to nodes in original dataset.
    Args:
        dataset: the input dataset
        num_nodes: the number of nodes
        num_features: the number of features
        num_edges: the number of edges
        num_classes: the number of classes
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

    # Number of OOD samples to generate
    num_sample_points = int(num_nodes * sampling_ratio)

    # Number of ID samples defined as points near the boundary
    num_boundary = int(num_nodes * boundary_ratio)

    # Number of boundaries used to generate outliers
    num_selected_boundaries = int(num_boundary * boundary_sampling_ratio)

    # Number of nearest neighbors to return
    k = min(len(dataset), k)

    print(f"\n{'Begin Generate Outliers':=^80}")
    print(f"Number of OOD samples to generate: {num_sample_points}")
    print(f"Number of ID samples defined as points near the boundary: {num_boundary}")
    print(f"Number of boundaries used to generate outliers: {num_selected_boundaries}")
    print(f"Number of nearest neighbors to return: {k}")
    print(f"Device for generating outliers: {device.type}")

    faiss_index = setup_faiss(num_features=num_features, device=device)

    # Randomly select a subset of points from the dataset
    random_index = np.random.choice(num_nodes, num_sample_points, replace=False)
    faiss_index.add(dataset[random_index].cpu().numpy())

    # Find the k most marginal elements of each element in the dataset, and then pick the top of them.
    # So that the most marginal elements of the entire dataset are selected.
    all_boundary_point_indices, max_distance = search_max_distance(
        dataset=dataset,
        index=faiss_index,
        k=k,
        top=num_boundary
    )

    # The farthest data points are selected, and a subset is randomly chosen from them.
    random_index = np.random.choice(num_boundary, num_selected_boundaries, replace=False)
    selected_boundary_point_indices = all_boundary_point_indices[random_index]

    # This is repeated n times, each time compensating for the length of the sample,
    # and the n results are pieced together, with each sample point contributing n features
    sampling_dataset = torch.cat([
        dataset[index].repeat(num_nodes // num_selected_boundaries, 1)
        for index in selected_boundary_point_indices
    ])

    # A multivariate Gaussian distribution with mean 0 and covariance matrix as identity matrix.
    try:
        gaussian_distribution = MultivariateNormal(
            loc=torch.zeros(num_features, device=device),
            covariance_matrix=torch.eye(num_features, device=device)
        )
    except NotImplementedError:
        candidate_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        gaussian_distribution = MultivariateNormal(
            loc=torch.zeros(num_features, device=candidate_device),
            covariance_matrix=torch.eye(num_features, device=candidate_device),
        )

    # Sampling from a Gaussian distribution as noise
    noises = gaussian_distribution.rsample(sample_shape=(sampling_dataset.shape[0],)).to(device)

    # Noise is added to the sampling points
    noise_cov = cov_mat * noises
    sampling_dataset += noise_cov

    sample_points = generate_sample_points(
        dataset=sampling_dataset,
        index=faiss_index,
        k=k,
        num_points=num_selected_boundaries,
        num_negative_samples=num_sample_points
    )

    faiss_index.reset()

    # The number of edges is determined by calculating the edge-to-point ratio.
    edge_node_radio = num_edges // num_nodes

    num_nodes = min(num_sample_points, len(sample_points))
    sample_edges = torch.randint(
        low=0,
        high=num_nodes - 1,
        size=(2, int(edge_node_radio * num_sample_points)),
        device=device
    )

    sample_labels = torch.randint(
        low=0,
        high=num_classes,
        size=(num_nodes,),
        device=device
    )

    print(f"Shape of OOD sample points: {sample_points.shape}")
    print(f"Shape of OOD sample edges: {sample_edges.shape}")
    print(f"Shape of OOD sample labels: {sample_labels.shape}")
    print(f"Time taken to generate outliers {round(time.time() - begin_time, 2)}s")
    print(f"{'End Generate Outliers':=^80}")

    return Outlier(
        sample_points=sample_points,
        sample_edges=sample_edges,
        sample_labels=sample_labels,
        all_boundary_point_indices=all_boundary_point_indices,
        selected_boundary_point_indices=selected_boundary_point_indices
    )


def setup_faiss(num_features: int, device: torch.device):
    faiss_index = faiss.IndexFlatL2(num_features)
    if "cuda" in device.type:
        resource = faiss.StandardGpuResources()
        device_index = device.index if device.index else 0
        # C/C++ prototypes are:
        # faiss::gpu::index_cpu_to_gpu(faiss::gpu::GpuResourcesProvider *,int,faiss::Index const *)
        # faiss::gpu::index_cpu_to_gpu(faiss::gpu::GpuResourcesProvider *,int,faiss::Index const *,faiss::gpu::GpuClonerOptions const *)
        faiss_index = faiss.index_cpu_to_gpu(provider=resource, device=device_index, index=faiss_index)
    return faiss_index


def search_max_distance(
        dataset: Tensor,
        index: IndexFlat,
        k: int,
        top: int
) -> tuple[Tensor, Tensor]:
    """
    Find the k neighbors of each element in the dataset,
    then select the farthest (Kth) neighbor of each element,
    and finally select the neighbors with the max distance neighbors from the Kth neighbors.

    Such as:
    k = 2, top = 2
    Elements   =>   Nearest neighbors  =>  Farthest neighbors  =>  Max distance neighbors
                        [[B, C],                [[C],
    [A, B, C]  =>        [C, A],       =>        [A],          =>  [C, A]
                         [A, B]]                 [B]]
    Args:
        dataset: the target of the search
        index: The index structure used for the search.
        k: The number of nearest neighbors to return.
        top: The number of points to return from the nearest neighbors.

    Returns: max_distance_index, max_distance_value

    """
    # Find the k nearest elements for each element and return a list of distances and indices
    dataset = dataset.cpu().numpy()
    if not dataset.flags.c_contiguous:
        dataset = dataset.copy(order='C')
    distance, output_index = index.search(dataset, k)
    # Take the last (farthest) element of each element
    k_th_distance = torch.tensor(distance[:, -1])
    # Find the farthest selected elements in the entire data set
    max_distance, max_distance_index = torch.topk(k_th_distance, top)
    return max_distance_index, max_distance


def generate_sample_points(
        dataset: Tensor,
        index: IndexFlat,
        k: int,
        num_points: int,
        num_negative_samples: int
) -> Tensor:
    """
    Generate negative samples by adding noise to the target points.

    Args:
        dataset: the target of the search. ([Batch, Features])
        index: The index structure used for the search.
        k: the number of nearest neighbors to return.
        num_points: The number of points to return from the nearest neighbors.
        num_negative_samples: the number of negative samples to generate.

    Returns: sample_points

    """
    distance, output_index = index.search(dataset.cpu().numpy(), k)
    # Find the last (farthest) neighbor of each element.
    k_th_distance = torch.tensor(distance[:, -1])
    # Find the maximum distance in the dataset.
    max_distance, max_distance_index = torch.topk(k_th_distance, num_points, dim=0)

    outliers = dataset[max_distance_index].repeat(repeats=(num_negative_samples // num_points, 1))

    # Add noise to the outliers
    beta = 0.05
    nosies = torch.cat([torch.rand_like(outliers) * -beta, torch.rand_like(outliers) * beta])
    num_noises = nosies.size(0)

    # Shuffle the order of the noise randomly
    shuffled_indices = torch.randperm(num_noises)
    nosies = nosies[shuffled_indices]
    nosies = nosies[:num_noises // 2, :]
    return outliers + nosies


if __name__ == '__main__':
    dataset = torch.rand(2000, 3)

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
