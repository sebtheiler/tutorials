from tensorflow.keras.layers import Conv2D, Activation, Concatenate, Conv2DTranspose, Input
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow_addons.layers import InstanceNormalization

img_rows, img_cols, channels = 256, 256, 3
weight_initializer = RandomNormal(stddev=0.02)

# "c7s1-k denotes a 7×7 Convolution-InstanceNorm-ReLU with k filters and stride 1"
def c7s1k(input, k, activation):
    block = Conv2D(k, (7, 7), padding='same', kernel_initializer=weight_initializer)(input)
    block = InstanceNormalization(axis=-1)(block)
    block = Activation(activation)(block)

    return block

# "dk denotes a 3×3 Convolution-InstanceNorm-ReLU with k filters and stride 2"
def dk(input, k):
    block = Conv2D(k, (3, 3), strides=2, padding='same', kernel_initializer=weight_initializer)(input)
    block = InstanceNormalization(axis=-1)(block)
    block = Activation('relu')(block)

    return block

# "Rk denotes a residual block that contains two 3×3 convolutional layers with k filters on each layer"
def Rk(input, k):
    block = Conv2D(k, (3, 3), padding='same', kernel_initializer=weight_initializer)(input)
    block = InstanceNormalization(axis=-1)(block)
    block = Activation('relu')(block)

    block = Conv2D(k, (3, 3), padding='same', kernel_initializer=weight_initializer)(block)
    block = InstanceNormalization(axis=-1)(block)

    return block + input

# "uk denotes a 3×3 fractional-strided-ConvolutionInstanceNorm-ReLU layer with k filters and stride ½"
def uk(input, k):
    # For the implementation: Conv2DTranspose(..., stride=2) = Conv2D(..., stride=0.5)
    block = Conv2DTranspose(k, (3, 3), strides=2, padding='same', kernel_initializer=weight_initializer)(input)
    block = InstanceNormalization(axis=-1)(block)
    block = Activation('relu')(block)

    return block

# c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, R256, R256, R256, u128, u64, c7s1-3
def generator(res_layers=9):
    gen_input = Input(shape=(img_rows, img_cols, channels))

    gen = c7s1k(gen_input, 64, 'relu')
    gen = dk(gen, 128)
    gen = dk(gen, 256)

    for _ in range(res_layers):
        gen = Rk(gen, 256)

    gen = uk(gen, 128)
    gen = uk(gen, 64)

    gen = c7s1k(gen, 3, 'tanh')

    return Model(gen_input, gen)
