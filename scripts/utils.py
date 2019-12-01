# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/7/10 22:39
   desc: the project
"""


def visualize_dmap(img, dmap):
    """

    :param img:
    :param dmap:
    :return:
    """
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(img)
    plt.subplot(2, 1, 2)
    plt.imshow(np.squeeze(dmap, axis=-1), cmap='gray')
    plt.show()


if __name__ == '__main__':
    from data import MallDataset
    x, y = MallDataset().get_img_data(0, size=224)
    visualize_dmap(x, y)