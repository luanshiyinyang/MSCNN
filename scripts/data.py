# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/5/23 12:31
   desc: the project
"""
from scipy.io import loadmat
import glob
import cv2
import numpy as np


class MallDataset(object):

    def __init__(self):
        self.filenames = sorted(glob.glob('../data/mall_dataset/frames/*.jpg'), key=lambda x: int(x[-8:-4]))

    def get_train_num(self):
        return int(len(self.filenames) * 0.8)

    def get_valid_num(self):
        return len(self.filenames) - int(len(self.filenames) * 0.8)

    def get_annotation(self):
        """
        读取2000个图片的注解，得到 每个图片的人数 和 每章图片的所有人坐标
        Annotation按照图片命名顺序
        :return:
        """
        mat_annotation = loadmat('../data/mall_dataset/mall_gt.mat')
        count_data, position_data = mat_annotation['count'], mat_annotation['frame'][0]
        return count_data, position_data

    def get_pixels(self, img, img_index, positions, size):
        """
        生成密度图，准备输入神经网络
        :param img
        :param img_index
        :param positions
        :param size 神经网络输入层图片大小
        """
        h, w = img.shape[0], img.shape[1]
        proportion_h, proportion_w = size / h, size / w  # 输入层需求与当前图片大小对比
        pixels = np.zeros((size, size))

        for p in positions[img_index][0][0][0]:
            # 取出每个人的坐标
            now_x, now_y = int(p[0] * proportion_w), int(p[1] * proportion_h)  # 按照输入层要求调整坐标位置
            if now_x >= size or now_y >= size:
                # 越界则向下取整
                print("Sorry skip the point, its index of all is {}".format(img_index))
            else:
                pixels[now_y, now_x] += 1
        pixels = cv2.GaussianBlur(pixels, (15, 15), 0)
        return pixels

    def get_img_data(self, index, size):
        """
        读取源文件图片
        :param index 图片下标
        :param size 神经网络输入层尺寸
        :return:
        """
        _, positions = self.get_annotation()
        img = cv2.imread(self.filenames[index])
        density_map = np.expand_dims(self.get_pixels(img, index, positions, size // 4), axis=-1)
        img = cv2.resize(img, (size, size)) / 255.

        return img, density_map

    def gen_train(self, batch_size, size):
        """
        生成数据生成器
        :param batch_size:
        :param size:
        :return:
        """
        _, position = self.get_annotation()
        index_all = list(range(int(len(self.filenames) * 0.8)))  # 取出所有训练数据下标，默认数据的前80%为训练集

        i, n = 0, len(index_all)
        if batch_size > n:
            raise Exception('Batch size {} is larger than the number of dataset {}!'.format(batch_size, n))

        while True:
            if i + batch_size >= n:
                np.random.shuffle(index_all)
                i = 0
                continue
            batch_x, batch_y = [], []
            for j in range(i, i + batch_size):
                x, y = self.get_img_data(index_all[j], size)
                batch_x.append(x)
                batch_y.append(y)
            i += batch_size
            yield np.array(batch_x), np.array(batch_y)

    def gen_valid(self, batch_size, size):
        """
        生成数据生成器
        :param batch_size:
        :param size:
        :return:
        """
        _, position = self.get_annotation()
        index_all = list(range(int(len(self.filenames) * 0.8), len(self.filenames)))

        i, n = 0, len(index_all)
        if batch_size > n:
            raise Exception('Batch size {} is larger than the number of dataset {}!'.format(batch_size, n))

        while True:
            if i + batch_size >= n:
                np.random.shuffle(index_all)
                i = 0
                continue
            batch_x, batch_y = [], []
            for j in range(i, i + batch_size):
                x, y = self.get_img_data(index_all[j], size)
                batch_x.append(x)
                batch_y.append(y)
            i += batch_size

            yield np.array(batch_x), np.array(batch_y)

    def gen_all(self, pic_size):
        """
        数据生成器
        :param pic_size:
        :return:
        """
        x_data = []
        y_data = []
        for i in range(len(self.filenames)):
            image, map_ = self.get_img_data(i, pic_size)
            x_data.append(image)
            y_data.append(map_)
        x_data, y_data = np.array(x_data), np.array(y_data)
        return x_data, y_data


class ShanghaitechDataset(object):

    def __init__(self, part='A'):
        if part == 'A':
            self.folder = '../data/ShanghaiTech/part_A/'
        else:
            self.folder = '../data/ShanghaiTech/part_B/'

    def get_annotation(self, folder, index):
        """
        读取图片注解
        :param folder 路径必须是part_A/train_data/这一步
        :param index: 图片索引,1开始
        :return:
        """
        mat_data = loadmat(folder + 'ground-truth/GT_IMG_{}.mat'.format(index))
        positions, count = mat_data['image_info'][0][0][0][0][0], mat_data['image_info'][0][0][0][0][1][0][0]
        return positions, count

    def get_pixels(self, folder, img, img_index, size):
        """
        生成密度图，准备输入神经网络
        :param folder 当前所在数据目录，该数据集目录较为复杂
        :param img 原始图像
        :param img_index 图片在当前目录下的图片序号，1开始
        :param size 目标图大小，按照模型为img的1/4
        """
        positions, _ = self.get_annotation(folder, img_index)
        h, w = img.shape[0], img.shape[1]
        proportion_h, proportion_w = size / h, size / w  # 输入层需求与当前图片大小对比
        pixels = np.zeros((size, size))

        for p in positions:
            # 取出每个人的坐标
            now_x, now_y = int(p[0] * proportion_w), int(p[1] * proportion_h)  # 按照输入层要求调整坐标位置
            if now_x >= size or now_y >= size:
                # 越界则向下取整
                pass
                # print("Sorry skip the point, its index of all is {}".format(img_index))
            else:
                pixels[now_y, now_x] += 1

        pixels = cv2.GaussianBlur(pixels, (15, 15), 0)
        return pixels

    def gen_train(self, batch_size, size):
        """
        获取训练数据
        :return:
        """
        folder = self.folder + 'train_data/'
        index_all = [i+1 for i in range(len(glob.glob(folder + 'images/*')))]

        i, n = 0, len(index_all)
        if batch_size > n:
            raise Exception('Batch size {} is larger than the number of dataset {}!'.format(batch_size, n))

        while True:
            if i + batch_size >= n:
                np.random.shuffle(index_all)
                i = 0
                continue
            batch_x, batch_y = [], []
            for j in range(i, i + batch_size):
                img = cv2.imread(folder + 'images/IMG_{}.jpg'.format(index_all[j]))
                density = np.expand_dims(self.get_pixels(folder, img, index_all[j], size // 4), axis=-1)
                img = cv2.resize(img, (size, size)) / 255.
                density = density.reshape([density.shape[0], density.shape[1], -1])
                batch_x.append(img)
                batch_y.append(density)
            i += batch_size
            yield np.array(batch_x), np.array(batch_y)

    def gen_valid(self, batch_size, size):
        """
        获取验证数据
        :return:
        """
        folder = self.folder + 'test_data/'
        index_all = [i + 1 for i in range(len(glob.glob(folder + 'images/*')))]

        i, n = 0, len(index_all)
        if batch_size > n:
            raise Exception('Batch size {} is larger than the number of dataset {}!'.format(batch_size, n))

        while True:
            if i + batch_size >= n:
                np.random.shuffle(index_all)
                i = 0
                continue
            batch_x, batch_y = [], []
            print("hello")
            for j in range(i, i + batch_size):
                img = cv2.imread(folder + 'images/IMG_{}.jpg'.format(index_all[j]))
                img = cv2.resize(img, (size, size)) / 255.
                density = np.expand_dims(self.get_pixels(folder, img, index_all[j], size // 4), axis=-1)
                density = density.reshape([density.shape[0], density.shape[1], -1])
                batch_x.append(img)
                batch_y.append(density)
            i += batch_size
            yield np.array(batch_x), np.array(batch_y)

    def get_train_num(self):
        return len(glob.glob(self.folder + 'train_data/' + 'images/*'))

    def get_valid_num(self):
        return len(glob.glob(self.folder + 'test_data/' + 'images/*'))


if __name__ == '__main__':
    MallDataset().gen_valid(16, 224)