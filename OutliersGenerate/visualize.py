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


def visualize_3D(dataset, boundary, outlier, title="visualize_3D"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        dataset[:, 0],
        dataset[:, 1],
        dataset[:, 2],
        c='tab:orange',
        label="dataset",
        alpha=0.5
    )
    ax.scatter(
        dataset[boundary, 0],
        dataset[boundary, 1],
        dataset[boundary, 2],
        c='tab:blue',
        label="boundary",
        alpha=0.5
    )
    ax.scatter(
        outlier[:, 0],
        outlier[:, 1],
        outlier[:, 2],
        c='tab:green',
        label="outlier",
        alpha=0.5
    )
    ax.title(title)
    ax.legend()
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
