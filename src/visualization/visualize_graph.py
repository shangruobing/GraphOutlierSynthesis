import sys
from os.path import dirname, abspath

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch_geometric.transforms as T
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

BASE_DIR = dirname(dirname(dirname(abspath(__file__))))
print(f"BASE_DIR:{BASE_DIR}")
sys.path.append(BASE_DIR)

from src.model.dataset import create_structure_manipulation_dataset, create_feature_interpolation_dataset, create_label_leave_out_dataset, create_knn_dataset


def visualize_dataset(data, title: str):
    G = to_networkx(data, to_undirected=True)

    plt.figure(figsize=(16, 16))
    pos = nx.spring_layout(G, seed=42)
    node_colors = data.y.numpy() if hasattr(data, 'y') else 'blue'

    nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=50,
        node_color=node_colors,
        edge_color="gray",
        alpha=0.6,
        cmap="tab10"
    )
    plt.title("Graph Visualization")
    plt.savefig(f"visualization/node_{title}.png")

    tsne = TSNE(n_components=2, random_state=42)
    node_embeddings_2d = tsne.fit_transform(data.x)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        node_embeddings_2d[:, 0],
        node_embeddings_2d[:, 1],
        c=data.y,
        cmap="tab10",
        s=15,
    )
    plt.colorbar(scatter, label="Node Classes")
    plt.title("Dataset Visualization")
    plt.xlabel("TSNE Dimension 1")
    plt.ylabel("TSNE Dimension 2")
    plt.savefig(f"visualization/tsne_{title}.png")


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    transform = T.NormalizeFeatures()
    dataset = Planetoid(root="./dataset/Planetoid", name="cora", split='public', transform=transform)
    data = dataset[0].to(device)
    visualize_dataset(data.to("cpu"), "origin")

    ood = create_structure_manipulation_dataset(data)
    visualize_dataset(ood, "structure_manipulation")

    ood = create_feature_interpolation_dataset(data)
    visualize_dataset(ood, "feature_interpolation")

    _, ood, _ = create_label_leave_out_dataset(data)
    visualize_dataset(ood, "label_leave_out")

    ood = create_knn_dataset(data=data, device=device).to("cpu")
    visualize_dataset(ood, "knn")


if __name__ == '__main__':
    main()
