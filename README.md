# Multi-scale Convolution Neural Networks for Crowd Counting

>密度图还要使用高斯滤波处理是因为在空间中计数时，每个人只占一个像素点导致最终得到的密度分布图特别稀疏，会导致模型收敛到全0状态。因此通过高斯处理后，密度图呈现出热力图的形式，一定程度上解决了稀疏问题。而且高斯处理后的密度图，总计数是不变的。


## 项目简介
复现论文[Multi-scale Convolution Neural Networks for Crowd Counting](https://arxiv.org/abs/1702.02359)。考虑到目前还没有具体写的比较完善的基于Keras（或者TensorFlow2）的复现，这对于迅速成型的顶层系统的构建不太方便。本项目基于Keras（Tensorflow后端），在多个数据集上进行训练测试，模型泛化能力强。


## 数据集下载
- ShanghaiTech Dataset
  - [下载地址](https://drive.google.com/open?id=1CW6PiAnLSWuUBX-2tVqQO5-1TDdilJB1)
- Mall Dataset
  - [下载地址](https://drive.google.com/open?id=170bssJjE_UbGeGSc_s2WHGBtbDAZRd7t)
- The_UCF_CC_50 Dataset
  - [下载地址](https://drive.google.com/open?id=1MwfTXFQUTx_sqw-g-D7TDOox1S88XYVN)
- 地址说明
  - 不提供数据集官方地址，数据集均放置在我的谷歌云盘，开启共享，无法翻墙的可以[邮箱](mailto:luanshiyinyang@gmail.com)联系我。
## 论文说明
- 针对神经网络近几年的发展以及现有的网络模型难以优化以及计算耗时，主要提出了multi-scale blob模块（类Inception结构）进行相关特征的提取。
- 作者主要提出了MSCNN的结构，该结构比起MCNN具有更好的处理能力及效果且参数量大幅度缩减，并且纵向对比了LBP+RR、MCNN+CCR、Zhang et al.和MCNN等模型。
- 论文文件
    - [PDF论文文件](/assets/1702.02359.pdf)
## 环境配置
- 基于Python3.6
- 需要第三方包已在[requirements](/requirements.txt)列出
	- 切换到requirements文件所在目录，执行命令`pip install -r requirements.txt`即可配置环境
- 脚本运行说明
	- 训练
		- 命令行执行
			- `python train.py -b 16`
		- 更详细的选项可以执行`python train.py -h`查看帮助
	- 测试
    	- 命令行执行
    		- `python test.py -s yes`
		- 更详细的选项可以执行`python test.py -h`查看帮助
## 模型构建
- 使用Keras的Function API构建模型
    - 代码
    	- ```python
			from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Activation, Dense
			from keras.layers.normalization import BatchNormalization
			from keras.models import Model


			def MSB(filter_num):
				def f(x):
					params = {
						'strides': 1,
						'activation': None,
						'padding': 'same'
					}
					x1 = Conv2D(filters=filter_num, kernel_size=(9, 9), **params)(x)
					x2 = Conv2D(filters=filter_num, kernel_size=(7, 7), **params)(x)
					x3 = Conv2D(filters=filter_num, kernel_size=(5, 5), **params)(x)
					x4 = Conv2D(filters=filter_num, kernel_size=(3, 3), **params)(x)
					x = concatenate([x1, x2, x3, x4])
					x = BatchNormalization()(x)
					return x
				return f


			def MSB_mini(filter_num):
				def f(x):
					params = {
						'strides': 1,
						'activation': None,
						'padding': 'same'
					}
					x2 = Conv2D(filters=filter_num, kernel_size=(7, 7), **params)(x)
					x3 = Conv2D(filters=filter_num, kernel_size=(5, 5), **params)(x)
					x4 = Conv2D(filters=filter_num, kernel_size=(3, 3), **params)(x)
					x = concatenate([x2, x3, x4])
					x = BatchNormalization()(x)
					return x
				return f


			def MSCNN(input_shape=(224, 224, 3)):
				"""
				模型构建
				本论文模型简单
				:param input_shape 输入图片尺寸
				:return:
				"""
				input_layer = Input(shape=input_shape)
				# block1
				x = Conv2D(filters=64, kernel_size=(9, 9), strides=1, padding='same', activation='relu')(input_layer)
				# block2
				x = MSB(4*16)(x)
				x = Activation('relu')(x)
				x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
				# block3
				x = MSB(4*32)(x)
				x = Activation('relu')(x)
				x = MSB(4*32)(x)
				x = Activation('relu')(x)
				x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

				x = MSB_mini(3*64)(x)
				x = Activation('relu')(x)
				x = MSB_mini(3*64)(x)
				x = Activation('relu')(x)

				x = Conv2D(1000, (1, 1), activation='relu')(x)

				x = Conv2D(1, (1, 1), activation='sigmoid')(x)
				x = Activation('relu')(x)

				model = Model(inputs=input_layer, outputs=x)
				return model
			```
    - 提示
        - **注意，输出层不能使用传统的Relu，会输出陷入“死区”，导致预测均为0值且loss确实在不断降低。**
	- 结构概念图
		- ![图片来自论文](/assets/mscnn.png)
	- 结构配置图
    	- ![图片来自论文](/assets/model.png)
## 模型训练
- 训练数据集
    - 主要在Mall dataset和ShanghaiTech上训练，其余数据集类似封装data loader即可。
- 训练效果展示（模型简单训练5轮）
    - mall_dataset
        - 对mall dataset上随机5张图片进行密度图预测，结果如下。
        - ![](/assets/rst.png)
    - shanghaitech
        - 提供训练API，未测试
## 补充说明
- 训练完成的预训练模型可以在我的drive云盘下载，下载后放置在models文件夹即可。
- 完整代码已经上传到我的Github，欢迎Star或者Fork。
- 如有错误，欢迎指正。
