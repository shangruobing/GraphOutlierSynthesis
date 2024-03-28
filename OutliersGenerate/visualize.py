"""
@author Ruobing Shang 2024-03-21 10:04
"""
import matplotlib.pyplot as plt


def visualize_2D(dataset, all_boundary, boundary, outlier, title="visualize_2D"):
    plt.figure(figsize=(10, 10))
    scatter_configs = [
        {
            "x": dataset[:, 0],
            "y": dataset[:, 1],
            "c": 'tab:orange',
            'label': "dataset",
            "alpha": 0.5
        },
        {
            "x": dataset[all_boundary, 0],
            "y": dataset[all_boundary, 1],
            "c": 'tab:blue',
            'label': "boundary",
            "alpha": 0.7
        },
        {
            "x": outlier[:, 0],
            "y": outlier[:, 1],
            "c": 'tab:green',
            'label': "outlier",
            "alpha": 0.5
        }
    ]
    for config in scatter_configs:
        plt.scatter(
            x=config["x"],
            y=config["y"],
            c=config["c"],
            label=config["label"],
            alpha=config["alpha"]
        )
    plt.scatter(
        dataset[boundary, 0],
        dataset[boundary, 1],
        label="selected boundary",
        alpha=0.7,
        s=200,
        marker='o',
        facecolor='none',
        edgecolor='tab:red',
        linestyle='dashed',
        linewidth=2
    )
    plt.title(title)
    plt.legend()
    plt.show()


def visualize_3D(dataset, all_boundary, boundary, outlier, title="visualize_3D"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter_configs = [
        {
            "x": dataset[:, 0],
            "y": dataset[:, 1],
            "z": dataset[:, 2],
            "c": 'tab:orange',
            'label': "dataset",
            "alpha": 0.5
        },
        {
            "x": dataset[all_boundary, 0],
            "y": dataset[all_boundary, 1],
            "z": dataset[all_boundary, 2],
            "c": 'tab:blue',
            'label': "boundary",
            "alpha": 0.7
        },
        {
            "x": outlier[:, 0],
            "y": outlier[:, 1],
            "z": outlier[:, 2],
            "c": 'tab:green',
            'label': "outlier",
            "alpha": 0.5
        }
    ]
    for config in scatter_configs:
        ax.scatter(
            config["x"],
            config["y"],
            config["z"],
            c=config["c"],
            label=config["label"],
            alpha=config["alpha"]
        )
    ax.scatter(
        dataset[boundary, 0],
        dataset[boundary, 1],
        dataset[boundary, 2],
        label="selected boundary",
        alpha=0.7,
        s=200,
        marker='o',
        facecolor='none',
        edgecolor='tab:red',
        linestyle='dashed',
        linewidth=2
    )
    plt.title(title)
    ax.legend()
    plt.show()


from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def visualize_TSNE(emb, color, epoch):
    fig = plt.figure(figsize=(6, 6), frameon=False)
    fig.suptitle(f'Epoch = {epoch}')

    # TSNE降为2维
    z = TSNE(n_components=2).fit_transform(emb.detach().cpu().numpy())
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0],
                z[:, 1],
                s=30,
                c=color.detach().cpu().numpy(),
                cmap="Set2")
    plt.show()


def visualize_classify(x, y):
    plt.scatter(range(len(x)), x, color='tab:red', label='Predicted', marker='x', alpha=0.7)
    plt.scatter(range(len(y)), y, color='tab:blue', label='True', marker='o', alpha=0.1)
    plt.xlabel('Sample Index')
    plt.ylabel('Label')
    plt.title('Predicted vs True Labels')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    import numpy as np

    dataset = np.random.rand(100, 2)
    all_boundary = np.arange(10)
    boundary = np.random.choice(all_boundary, size=4, replace=False)
    outlier = np.random.rand(30, 2)

    visualize_2D(
        dataset,
        all_boundary,
        boundary,
        outlier,
        title="visualize_2D"
    )

    import torch

    x = torch.randn(100)
    y = torch.randn(100)
    visualize_classify(x, y)
