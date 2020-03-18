from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Activation, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def MSB(filter_num):
    def f(x):
        params = {
            'strides': 1,
            'activation': 'relu',
            'padding': 'same',
            'kernel_regularizer': l2(5e-4)
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
            'activation': 'relu',
            'padding': 'same',
            'kernel_regularizer': l2(5e-4)
        }
        x2 = Conv2D(filters=filter_num, kernel_size=(7, 7), **params)(x)
        x3 = Conv2D(filters=filter_num, kernel_size=(5, 5), **params)(x)
        x4 = Conv2D(filters=filter_num, kernel_size=(3, 3), **params)(x)
        x = concatenate([x2, x3, x4])
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    return f


def MSCNN(input_shape=(224, 224, 3)):
    """
    模型构建
    本论文模型简单
    :param input_shape 输入图片尺寸
    :return:
    """
    input_tensor = Input(shape=input_shape)
    # block1
    x = Conv2D(filters=64, kernel_size=(9, 9), strides=1, padding='same', activation='relu')(input_tensor)
    # block2
    x = MSB(4*16)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    # block3
    x = MSB(4*32)(x)
    x = MSB(4*32)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = MSB_mini(3*64)(x)
    x = MSB_mini(3*64)(x)

    x = Conv2D(1000, (1, 1), activation='relu', kernel_regularizer=l2(5e-4))(x)

    x = Conv2D(1, (1, 1))(x)
    x = Activation('sigmoid')(x)
    x = Activation('relu')(x)

    model = Model(inputs=input_tensor, outputs=x)
    return model


if __name__ == '__main__':
    mscnn = MSCNN((224, 224, 3))
    print(mscnn.summary())

