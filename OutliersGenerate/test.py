from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

import numpy as np


# def plt2arr(fig):
#     rgb_str = fig.canvas.buffer_rgba()  # 将图形对象转换为RGB字符串表示
#     (w, h) = fig.canvas.get_width_height()  # 获取图形对象的宽度和高度
#     rgba_arr = np.frombuffer(rgb_str, dtype=np.uint8).reshape((h, w, 3))  # 使用NumPy从字符串中创建一个三维数组（高度 x 宽度 x 3通道）
#     return rgba_arr  # 返回转换后的NumPy数组


def visualize(emb, color, epoch):
    fig = plt.figure(figsize=(6, 6), frameon=False)
    fig.suptitle(f'Epoch = {epoch}')

    # TSNE降为2维
    z = TSNE(n_components=2).fit_transform(emb.detach().cpu().numpy())
    # 散点图
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0],
                z[:, 1],
                s=30,
                c=color.detach().cpu().numpy(),
                cmap="Set2")
    # 强制显示
    plt.show()
    # fig.canvas.draw()
    # return plt2arr(fig)


def line(x, y):
    plt.scatter(range(len(x)), x, color='r', label='Predicted', marker='x')
    plt.scatter(range(len(y)), y, color='b', label='True', marker='o')

    plt.xlabel('Sample Index')
    plt.ylabel('Label')
    plt.title('Predicted vs True Labels')

    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    import torch

    # predict = [1, 0, 1, 0, 1]
    # label = [0, 1, 0, 1, 0]
    #
    # #
    # x = torch.randn(100)
    # print(x)
    # y = torch.randn(100)
    # line(x, y)
    # y = torch.randint(
    #     low=0,
    #     high=3,
    #     size=(len(outputs),),
    # )
    #
    # visualize(outputs, color=y, epoch=1)
