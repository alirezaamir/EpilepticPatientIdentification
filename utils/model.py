#  MIT License
#
#  Copyright (c)2020 Embedded System Lab, EPFL
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.layers import Add, Concatenate
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import MaxPooling1D, AveragePooling1D

vector_len = 4136
seiz_len = 2048
non_seiz_len = 2048


def residual_block(in_cnv, stage: str, dropout=True):
    bn1 = BatchNormalization(name='BN1_' + stage)(in_cnv)
    relu1 = ReLU(name='ReLU1_' + stage)(bn1)
    drop = Dropout(rate = 0.1, name='Drop_' + stage)(relu1)

    cnv2 = Conv1D(32, kernel_size=5, padding='same', name='Conv2_' + stage)(drop if dropout else relu1)
    bn2 = BatchNormalization(name='BN2_' + stage)(cnv2)
    relu2 = ReLU(name='ReLU2_' + stage)(bn2)

    cnv3 = Conv1D(32, kernel_size=5, padding='same', name='Conv3_' + stage)(relu2)
    add = Add(name='Add_' + stage)([in_cnv, cnv3])
    pool = MaxPooling1D(pool_size=5, strides=2, name='Pool_' + stage)(add)
    return pool


def my_residual_block(in_cnv, stage: str, filter_size, dropout=True, stride=1):
    bn1 = BatchNormalization(name='BN1_' + stage)(in_cnv)
    relu1 = ReLU(name='ReLU1_' + stage)(bn1)
    drop = Dropout(rate = 0.1, name='Drop_' + stage)(relu1)

    cnv2 = Conv1D(filter_size, strides=stride, kernel_size=5, padding='same', name='Conv2_' + stage)(drop if dropout else relu1)
    bn2 = BatchNormalization(name='BN2_' + stage)(cnv2)
    relu2 = ReLU(name='ReLU2_' + stage)(bn2)

    cnv3 = Conv1D(filter_size, strides=1, kernel_size=5, padding='same', name='Conv3_' + stage)(relu2)
    size_changed = (in_cnv.shape[2] != filter_size)
    if size_changed:
        shortcut = Conv1D(filter_size, kernel_size=1, strides=stride, name='Shortcut_'+stage, padding='same')(in_cnv)
        add = Add(name='Add_' + stage)([shortcut, cnv3])
    elif stride>1:
        pool = MaxPooling1D(pool_size=5, strides=stride, name='Pool_'+stage, padding='same')(in_cnv)
        add = Add(name='Add_' + stage)([pool, cnv3])
    else:
        add = Add(name='Add_' + stage)([in_cnv, cnv3])
    return add


def get_model(output_number = 2, dropout = True):
    signal_input = Input(shape=(vector_len, 1))
    x1_cnv = Conv1D(32, kernel_size=7, strides=2, padding='same')(signal_input)
    x1_bn = BatchNormalization()(x1_cnv)
    x1_relu = ReLU()(x1_bn)

    x1_cnv = Conv1D(32, kernel_size=5, padding='same')(x1_relu)
    res1 = residual_block(x1_cnv, '1', dropout)
    res2 = residual_block(res1, '2', dropout)
    res3 = residual_block(res2, '3', dropout)
    res4 = residual_block(res3, '4', dropout)
    res5 = residual_block(res4, '10', dropout)

    bn5 = BatchNormalization(name='BN_5')(res5)
    relu5 = ReLU(name='ReLU_5')(bn5)
    flatten = Flatten(name='Flat_5')(relu5)
    drop1 = Dropout(rate=0.5)(flatten)
    dense1 = Dense(64, activation='relu', name='Dense1_5')(drop1)
    drop2 = Dropout(rate=0.5)(dense1)
    dense2 = Dense(output_number, activation='softmax', name='Dense2_5')(drop2)

    model = Model(inputs=signal_input, outputs=dense2)
    return model


def get_resnet_model(output_number=2, dropout=True):
    signal_input = Input(shape=(vector_len, 1))
    x1_cnv = Conv1D(32, kernel_size=5, strides=2, padding='same')(signal_input)
    res1 = my_residual_block(x1_cnv, '1', filter_size=64, dropout=dropout, stride=2)
    res2 = my_residual_block(res1, '2', filter_size=64, dropout=dropout, stride=2)
    res3 = my_residual_block(res2, '3', filter_size=128, dropout=dropout, stride=2)
    # res4 = my_residual_block(res3, '4', filter_size=128, dropout=dropout, stride=2)

    bn5 = BatchNormalization(name='BN_5')(res3)
    relu5 = ReLU(name='ReLU_5')(bn5)
    avg = AveragePooling1D(pool_size=5)(relu5)
    flatten = Flatten(name='Flat_5')(avg)
    drop1 = Dropout(rate=0.5)(flatten)
    dense1 = Dense(256, activation='relu', name='Dense1_5')(drop1)
    drop2 = Dropout(rate=0.5)(dense1)
    dense2 = Dense(output_number, activation='softmax', name='Dense2_5')(drop2)

    model = Model(inputs=signal_input, outputs=dense2)
    return model


def get_vgg_block(in_cnv, stage: str, filters: int):
    x_cnv = Conv1D(filters, kernel_size=3, strides=1, padding='same', name='CNV_'+stage)(in_cnv)
    x_bn = BatchNormalization(name='BN_'+stage)(x_cnv)
    x_relu = ReLU(name='ReLU_'+stage)(x_bn)
    return x_relu


def get_vgg_model(output_number=2, dropout= True):
    signal_input = Input(shape=(vector_len, 1))
    x1 = get_vgg_block(signal_input, '1', 32)
    x2 = get_vgg_block(x1, '2', 32)
    pool1 = MaxPooling1D(pool_size=2, strides=2, name='Pool_1')(x2)

    x3 = get_vgg_block(pool1, '3', 64)
    x4 = get_vgg_block(x3, '4', 64)
    pool2 = MaxPooling1D(pool_size=2, strides=2, name='Pool_2')(x4)

    x5 = get_vgg_block(pool2, '5', 128)
    x6 = get_vgg_block(x5, '6', 128)
    pool3 = MaxPooling1D(pool_size=2, strides=2, name='Pool_3')(x6)

    x7 = get_vgg_block(pool3, '7', 256)
    x8 = get_vgg_block(x7, '8', 256)
    x9 = get_vgg_block(x8, '9', 256)
    pool4 = MaxPooling1D(pool_size=2, strides=2, name='Pool_4')(x9)

    x10 = get_vgg_block(pool4, '10', 256)
    x11 = get_vgg_block(x10, '11', 256)
    x12 = get_vgg_block(x11, '12', 256)
    pool5 = MaxPooling1D(pool_size=2, strides=2, name='Pool_5')(x12)

    flatten = Flatten(name='Flat_5')(pool5)
    drop1 = Dropout(rate=0.5)(flatten)
    dense1 = Dense(2048, activation='relu', name='Dense1')(drop1 if dropout else flatten)
    drop2 = Dropout(rate=0.5)(dense1)
    dense2 = Dense(1024, activation='relu', name='Dense2')(drop2 if dropout else dense1)
    drop3 = Dropout(rate=0.5)(dense2)
    dense3 = Dense(output_number, activation='softmax', name='Dense3')(drop3 if dropout else drop3)

    model = Model(inputs=signal_input, outputs=dense3)
    return model


def con_bn_relu(in_cnv, stage):
    cnv = Conv1D(32, kernel_size=5, padding='same', name='Conv_' + stage)(in_cnv)
    bn = BatchNormalization(name='BN_' + stage)(cnv)
    relu = ReLU(name='ReLU_' + stage)(bn)
    return relu


def get_branch(input_signal, name):
    x1_cnv = Conv1D(32, kernel_size=7, strides=2, padding='same')(input_signal)
    x1_bn = BatchNormalization()(x1_cnv)
    x1_relu = ReLU()(x1_bn)
    x2_cnv = con_bn_relu(x1_relu, name+'2')
    drop2 = Dropout(0.5, name='Drop2_'+name)(x2_cnv)
    x3_cnv = con_bn_relu(drop2, name+'3')
    pool3 = MaxPooling1D(pool_size=5, strides=2, name='Pool3_'+name)(x3_cnv)
    x4_cnv = con_bn_relu(pool3, name+'4')
    drop4 = Dropout(0.5, name='Drop4_'+name)(x4_cnv)
    x5_cnv = con_bn_relu(drop4, name+'5')
    pool5 = MaxPooling1D(pool_size=5, strides=2, name='Pool5_'+name)(x5_cnv)
    return pool5


def get_relation_model():
    seiz_input = Input(shape=(seiz_len, 1))
    seiz_branch = get_branch(seiz_input, 'seiz')

    non_seiz_input = Input(shape=(seiz_len, 1))
    non_seiz_branch = get_branch(non_seiz_input, 'non_seiz')

    concatenation = Concatenate(axis=2)([seiz_branch, non_seiz_branch])
    x6_cnv = con_bn_relu(concatenation, '6')
    pool6 = MaxPooling1D(pool_size=5, strides=2, name='Pool6')(x6_cnv)
    x7_cnv = con_bn_relu(pool6, '7')
    flatten = Flatten(name='Flat_5')(x7_cnv)
    dense7 = Dense(64, activation='relu', name='Dense1_5')(flatten)
    dense8 = Dense(2, activation='softmax', name='Dense2_5')(dense7)

    model = Model(inputs=[seiz_input, non_seiz_input], outputs=dense8)
    return model


def plot(model: Model, name: str):
    plot_model(model, to_file='outputs/' + name + '.png', show_shapes=True, show_layer_names=False)


if __name__ == '__main__':
    model = get_resnet_model(2, True)
    plot(model, 'My_ResNet_like')
