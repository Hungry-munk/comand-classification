import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.layers import Input, Activation, BatchNormalization
from tensorflow.keras.models import Model
from configs import Configs as C 

# get configs
c = C()

def conv_bn_pool_block(inputs, filters, conv_kernel, conv_padding, block_num, activation='relu', pooling=True, batch_norm=True, pool_kernel=(2, 2)):
    x = Conv2D(filters, conv_kernel, padding=conv_padding, name=f'conv {block_num}', kernel_initializer='he_normal')(inputs)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    if pooling:
        x = MaxPooling2D(pool_size=pool_kernel, name=f'max{block_num}')(x)
    return x

def build_CNN(input_shape, num_classes, main_activation = 'relu'):
    input_tensor = Input(input_shape)
    x = input_tensor

    x = conv_bn_pool_block(x , 32, (3, 3), 'same', 1, main_activation, True, True, (2, 2))
    x = conv_bn_pool_block(x, 64, (3, 3), 'same',2 , main_activation, True, True, (2, 2))
    
    x = Flatten()(x)

    x = Dense(128, activation = main_activation)(x)
    x = Dropout(0.3)(x)

    output_tensor = Dense(num_classes, activation = 'softmax')(x)

    return Model(inputs = input_tensor, outputs = output_tensor)