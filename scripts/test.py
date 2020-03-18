from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os
import random
import glob
import cv2
import numpy as np
from model import MSCNN
import scipy.io as sio


def parse_params():
    """
    解析命令行参数
    :return:
    """
    ap = ArgumentParser()
    ap.add_argument('-s', '--show', default='yes', help='if show test result map')
    args_ = ap.parse_args()
    args_ = vars(args_)
    return args_


def get_samples_malldataset(num):
    """
    获取测试图片
    :return:
    """
    def get_annotation():
        """
        读取2000个图片的注解，得到 每个图片的人数 和 每章图片的所有人坐标
        Annotation按照图片命名顺序
        :return:
        """
        mat_annotation = sio.loadmat('../data/mall_dataset/mall_gt.mat')
        count_data, position_data = mat_annotation['count'], mat_annotation['frame'][0]
        return count_data, position_data
    counts_true, _ = get_annotation()
    datasize = len(glob.glob('../data/mall_dataset/frames/*.jpg'))
    # 从验证集选取测试图片
    samples_index = random.sample([i for i in range(int(0.8*datasize), datasize)], num)
    samples = [glob.glob('../data/mall_dataset/frames/*.jpg')[i] for i in samples_index]
    images = []
    counts = []
    for i in range(num):
        filename = samples[i]
        print(filename)
        print(samples_index[i])
        img = cv2.resize(cv2.imread(filename), (224, 224)) / 255.
        img = np.expand_dims(img, axis=0)
        images.append(img)
        counts.append(counts_true[samples_index[i]])
    return images, counts


def get_samples_shanghaitech(num):
    """
    获取测试图片
    :return:
    """
    def get_annotation(index):
        """
        读取2000个图片的注解，得到 每个图片的人数 和 每章图片的所有人坐标
        Annotation按照图片命名顺序
        :return:
        """
        mat_data = sio.loadmat('../data/ShanghaiTech/part_A_final/test_data/ground_truth/GT_IMG_{}.mat'.format(index))
        position_data, count_data = mat_data['image_info'][0][0][0][0][0], mat_data['image_info'][0][0][0][0][1][0][0]
        return count_data, position_data

    datasize = len(glob.glob('../data/ShanghaiTech/part_A_final/test_data/images/*.jpg'))
    # 从验证集选取测试图片
    samples_index = random.sample([i for i in range(datasize)], num)
    samples = ['../data/ShanghaiTech/part_A_final/test_data/images/IMG_{}.jpg'.format(i) for i in samples_index]
    images = []
    counts = []
    for i in range(num):
        filename = samples[i]
        img = cv2.resize(cv2.imread(filename), (224, 224)) / 255.
        img = np.expand_dims(img, axis=0)
        images.append(img)
        counts.append(get_annotation(samples_index[i])[0])
    return images, counts


def plot_sample(raw_images, maps, counts, true_counts):
    """
    演示测试的5个图片
    :return:
    """
    plt.figure(figsize=(15, 9))
    for i in range(len(maps)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(np.squeeze(raw_images[i], axis=0))
        plt.title('people true num {}'.format(int(true_counts[i])))
        plt.subplot(2, 5, i + 1 + 5)
        plt.imshow(maps[i][0])
        plt.title('people pred num {}'.format(counts[i]))
    plt.savefig('../results/rst.png')


def save_result(raw_images, maps, counts, args_, true_counts):
    """
    保存map图
    :return:
    """
    if not os.path.exists('../results'):
        os.mkdir('../results')
    # for i in range(len(maps)):
    #     cv2.imwrite('../results/sample_{}.jpg'.format(i), maps[i])
    if args_['show'] == 'yes':
        plot_sample(raw_images, maps, counts, true_counts)


def test(args_):
    """
    测试模型效果
    :param args_:
    :return:
    """
    model = MSCNN((224, 224, 3))
    if os.path.exists('../models/best_model_weights.h5'):
        model.load_weights('../models/best_model_weights.h5')
        samples, true_counts = get_samples_shanghaitech(5)
        maps = []
        counts = []
        for sample in samples:
            dmap = np.squeeze(model.predict(sample), axis=-1)
            dmap = cv2.GaussianBlur(dmap, (15, 15), 0)
            counts.append(int(np.sum(dmap)))
            maps.append(dmap)
        save_result(samples, maps, counts, args_, true_counts)
    else:
        print("Sorry, cannot find model file in root_path/models/, please download my model or train your model")


if __name__ == '__main__':
    args = parse_params()
    test(args)