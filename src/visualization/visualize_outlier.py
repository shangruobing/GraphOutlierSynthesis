import sys
from os.path import abspath, dirname

import torch

BASE_DIR = dirname(dirname(dirname(abspath(__file__))))
print(f"BASE_DIR:{BASE_DIR}")
sys.path.append(BASE_DIR)

from src.visualization.visualize import visualize_2D, visualize_3D
from src.outlier.knn import generate_outliers


def visualize_2d():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = torch.rand(2000, 2)
    num_nodes, num_features = dataset.shape[0], dataset.shape[1]
    num_edges = 10
    outliers = generate_outliers(
        dataset=dataset,
        num_nodes=num_nodes,
        num_features=num_features,
        num_edges=num_edges,
        device=device
    )
    visualize_2D(
        dataset=dataset.cpu(),
        all_boundary=outliers.all_boundary_point_indices,
        boundary=outliers.selected_boundary_point_indices,
        outlier=outliers.sample_points.cpu()
    )


def visualize_3d():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = torch.rand(2000, 3)
    num_nodes, num_features = dataset.shape[0], dataset.shape[1]
    num_edges = 10
    outliers = generate_outliers(
        dataset=dataset,
        num_nodes=num_nodes,
        num_features=num_features,
        num_edges=num_edges,
        device=device
    )
    visualize_3D(
        dataset=dataset.cpu(),
        all_boundary=outliers.all_boundary_point_indices,
        boundary=outliers.selected_boundary_point_indices,
        outlier=outliers.sample_points.cpu()
    )


def main():
    visualize_2d()
    # visualize_3d()


if __name__ == '__main__':
    main()
