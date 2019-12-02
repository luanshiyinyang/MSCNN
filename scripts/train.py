# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/5/24 0:40
   desc: the project
"""
from argparse import ArgumentParser
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from model import MSCNN
from data import MallDataset, ShanghaitechDataset
import os
import warnings
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6 # 每个GPU现存上届控制在60%以内
session = tf.Session(config=config)
KTF.set_session(session )


def parse_command_params():
    """
    解析命令行参数
    :return:
    """
    parser = ArgumentParser()
    parser.add_argument('-e', '--epochs', default=50, help='how many epochs to fit')
    parser.add_argument('-v', '--show', default='yes', help='if show training log')
    parser.add_argument('-b', '--batch', default=16, help='batch size of train')
    parser.add_argument('-d', '--dataset', default='shanghaitechdataset', help='which dataset to train')
    parser.add_argument('-p', '--pretrained', default='no', help='load your pretrained model in folder root/models')
    args_ = parser.parse_args()
    args_ = vars(args_)
    return args_


def get_callbacks():
    """
    设置部分回调
    :return:
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-7, verbose=True)
    if not os.path.exists('../models'):
        os.mkdir('../models')
    model_checkpoint = ModelCheckpoint('../models/best_model_weights.h5', monitor='val_loss',
                                       verbose=True, save_best_only=True, save_weights_only=True)
    callbacks = [early_stopping, reduce_lr, model_checkpoint]
    return callbacks


def train(args_):
    """
    进行训练
    :return:
    """
    model = MSCNN((224, 224, 3))
    model.compile(optimizer=Adam(lr=3e-4), loss='mse')
    # load pretrained model
    if args_['pretrained'] == 'yes':
        model.load_weights('../models/best_model_weights.h5')
        print("load model from ../models/")

    callbacks = get_callbacks()

    # 流式读取，一个batch读入内存
    batch_size = int(args_['batch'])
    if args_['dataset'] == 'malldataset':
        model.fit_generator(MallDataset().gen_train(batch_size, 224),
                            steps_per_epoch=MallDataset().get_train_num() // batch_size,
                            validation_data=MallDataset().gen_valid(batch_size, 224),
                            validation_steps=MallDataset().get_valid_num() // batch_size,
                            epochs=int(args_['epochs']),
                            callbacks=callbacks)
    elif args_['dataset'] == 'shanghaitechdataset':
        model.fit_generator(ShanghaitechDataset().gen_train(batch_size, 224),
                            steps_per_epoch=ShanghaitechDataset().get_train_num() // batch_size,
                            validation_data=ShanghaitechDataset().gen_valid(batch_size, 224),
                            validation_steps=ShanghaitechDataset().get_valid_num() // batch_size,
                            epochs=int(args_['epochs']),
                            callbacks=callbacks)
    else:
        print('not support this dataset')


if __name__ == '__main__':
    args = parse_command_params()
    train(args)
