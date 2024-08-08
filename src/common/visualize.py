import matplotlib.pyplot as plt


def visualize_2D(dataset, all_boundary, boundary, outlier, title="2D Visualization"):
    plt.figure(figsize=(10, 10))
    scatter_configs = [
        {
            "x": dataset[:, 0],
            "y": dataset[:, 1],
            "c": 'tab:orange',
            'label': "Dataset",
            "alpha": 0.5
        },
        {
            "x": dataset[all_boundary, 0],
            "y": dataset[all_boundary, 1],
            "c": 'tab:blue',
            'label': "Boundary",
            "alpha": 0.7
        },
        {
            "x": outlier[:, 0],
            "y": outlier[:, 1],
            "c": 'tab:green',
            'label': "Outlier",
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
        label="Selected Boundaries",
        alpha=0.7,
        s=100,
        marker='o',
        facecolor='none',
        edgecolor='tab:red',
        linestyle='dashed',
        linewidth=2
    )
    plt.title(title, fontsize=16, fontweight='bold', pad=8)
    plt.xlabel('X Axis', fontsize=12, labelpad=4)
    plt.ylabel('Y Axis', fontsize=12, labelpad=4)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    plt.savefig("visualize_2D.png", dpi=300)
    plt.show()


def visualize_3D(dataset, all_boundary, boundary, outlier, title="3D Visualization"):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter_configs = [
        {
            "x": dataset[:, 0],
            "y": dataset[:, 1],
            "z": dataset[:, 2],
            "c": 'tab:orange',
            'label': "Dataset",
            "alpha": 0.5,
        },
        {
            "x": dataset[all_boundary, 0],
            "y": dataset[all_boundary, 1],
            "z": dataset[all_boundary, 2],
            "c": 'tab:blue',
            'label': "Boundary",
            "alpha": 0.7,
        },
        {
            "x": outlier[:, 0],
            "y": outlier[:, 1],
            "z": outlier[:, 2],
            "c": 'tab:green',
            'label': "Outlier",
            "alpha": 0.5,
        }
    ]

    for config in scatter_configs:
        ax.scatter(
            config["x"],
            config["y"],
            config["z"],
            c=config["c"],
            label=config["label"],
            alpha=config["alpha"],
            marker='o'
        )

    ax.scatter(
        dataset[boundary, 0],
        dataset[boundary, 1],
        dataset[boundary, 2],
        label="Selected Boundaries",
        alpha=0.7,
        s=100,
        marker='o',
        facecolor='none',
        edgecolor='tab:red',
        linestyle="dashed",
        linewidth=2
    )

    ax.set_title(title, fontsize=16, fontweight='bold', pad=8)
    ax.set_xlabel('X Axis', fontsize=12, labelpad=4)
    ax.set_ylabel('Y Axis', fontsize=12, labelpad=4)
    ax.set_zlabel('Z Axis', fontsize=12, labelpad=4)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    fig.savefig("visualize_3D.png", dpi=300)
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
