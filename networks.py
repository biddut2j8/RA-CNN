import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Conv2D, Lambda,MaxPooling2D, concatenate, UpSampling2D,Add, BatchNormalization
from tensorflow.keras import Model, layers

def ifft_layer(kspace):
    real = Lambda(lambda kspace : kspace[:,:,:,0])(kspace)
    imag = Lambda(lambda kspace : kspace[:,:,:,1])(kspace)
    kspace_complex = tf.complex(real,imag)
    rec1 = tf.abs(tf.signal.ifft2d(kspace_complex))
    rec1 = tf.expand_dims(rec1, -1)
    return rec1

def nrmse(y_true, y_pred):
    denom = K.sqrt(K.mean(K.square(y_true), axis=(1,2,3)))
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=(1,2,3)))\
    /denom


def gating_signal(input, out_size):
    x = layers.Conv2D(out_size, (1, 1),activation= 'relu', padding='same')(input)
    return x

def repeat_elem(tensor, rep):
     return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)


def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

# Getting the x signal to the same shape as the gating signal
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

# Getting the gating signal to the same number of filters as the inter_shape
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    y = layers.multiply([upsample_psi, x])

    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    #result_bn = layers.BatchNormalization()(result)
    return result
'''
48= 64
64= 128
128=256
256= 512
'''


def wnet(mu1, sigma1, mu2, sigma2, H=256, W=256, channels=2, kshape=(3, 3), kshape2=(3, 3)):
    inputs = Input(shape=(H, W, channels))

    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)

    #upsampling layers
    gating_0 = gating_signal(conv4, 128)
    att_0 = attention_block(conv3, gating_0, 128)
    up_0 = layers.UpSampling2D(size=(2, 2))(conv4)
    concate_0 = layers.concatenate([att_0, up_0], axis=-1)

    #up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(concate_0)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)

    gating_1 = gating_signal(conv5, 64)
    att_1 = attention_block(conv2, gating_1, 64)
    up_1 = layers.UpSampling2D(size=(2, 2))(conv5)
    concate_1 = layers.concatenate([att_1, up_1], axis=-1)

    #up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(concate_1)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)

    gating_2 = gating_signal(conv6, 48)
    att_2 = attention_block(conv1, gating_2, 48)
    up_2 = layers.UpSampling2D(size=(2, 2))(conv6)
    concate_2 = layers.concatenate([att_2, up_2], axis=-1)

    #up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(concate_2)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)

    conv8 = Conv2D(2, (1, 1), activation='linear')(conv7)
    res1 = Add()([conv8, inputs])
    res1_scaled = Lambda(lambda res1: (res1 * sigma1 + mu1))(res1)

    rec1 = Lambda(ifft_layer)(res1_scaled)
    rec1_norm = Lambda(lambda rec1: (rec1 - mu2) / sigma2)(rec1)

    conv9 = Conv2D(48, kshape2, activation='relu', padding='same')(rec1_norm)
    conv9 = Conv2D(48, kshape2, activation='relu', padding='same')(conv9)
    conv9 = Conv2D(48, kshape2, activation='relu', padding='same')(conv9)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv9)

    conv10 = Conv2D(64, kshape2, activation='relu', padding='same')(pool4)
    conv10 = Conv2D(64, kshape2, activation='relu', padding='same')(conv10)
    conv10 = Conv2D(64, kshape2, activation='relu', padding='same')(conv10)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv10)

    conv11 = Conv2D(128, kshape2, activation='relu', padding='same')(pool5)
    conv11 = Conv2D(128, kshape2, activation='relu', padding='same')(conv11)
    conv11 = Conv2D(128, kshape2, activation='relu', padding='same')(conv11)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv11)

    conv12 = Conv2D(256, kshape2, activation='relu', padding='same')(pool6)
    conv12 = Conv2D(256, kshape2, activation='relu', padding='same')(conv12)
    conv12 = Conv2D(256, kshape2, activation='relu', padding='same')(conv12)

    # upsampling layers
    gating_00 = gating_signal(conv12, 128)
    att_00 = attention_block(conv11, gating_00, 128)
    up_00 = layers.UpSampling2D(size=(2, 2))(conv12)
    concate_00 = layers.concatenate([att_00, up_00], axis=-1)

    #up4 = concatenate([UpSampling2D(size=(2, 2))(conv12), conv11], axis=-1)
    conv13 = Conv2D(128, kshape2, activation='relu', padding='same')(concate_00)
    conv13 = Conv2D(128, kshape2, activation='relu', padding='same')(conv13)
    conv13 = Conv2D(128, kshape2, activation='relu', padding='same')(conv13)

    gating_11 = gating_signal(conv13, 64)
    att_11 = attention_block(conv10, gating_11, 64)
    up_11 = layers.UpSampling2D(size=(2, 2))(conv13)
    concate_11 = layers.concatenate([att_11, up_11], axis=-1)

    #up5 = concatenate([UpSampling2D(size=(2, 2))(conv13), conv10], axis=-1)
    conv14 = Conv2D(64, kshape2, activation='relu', padding='same')(concate_11)
    conv14 = Conv2D(64, kshape2, activation='relu', padding='same')(conv14)
    conv14 = Conv2D(64, kshape2, activation='relu', padding='same')(conv14)

    gating_22 = gating_signal(conv14, 48)
    att_22 = attention_block(conv9, gating_22, 48)
    up_22 = layers.UpSampling2D(size=(2, 2))(conv14)
    concate_22 = layers.concatenate([att_22, up_22], axis=-1)

    #up6 = concatenate([UpSampling2D(size=(2, 2))(conv14), conv9], axis=-1)
    conv15 = Conv2D(48, kshape2, activation='relu', padding='same')(concate_22)
    conv15 = Conv2D(48, kshape2, activation='relu', padding='same')(conv15)
    conv15 = Conv2D(48, kshape2, activation='relu', padding='same')(conv15)

    out2 = Conv2D(1, (1, 1), activation='linear')(conv15)
    rec2 = Lambda(ifft_layer)(inputs)
    out = Add()([out2, rec2])
    model = Model(inputs=inputs, outputs=[res1_scaled, out])
    return model

if __name__ == "__main__":
    import numpy as np
    stats = np.load('../data/stats_fs_unet_norm_25.npy')
    model = wnet(stats[0],stats[1],stats[2],stats[3],kshape = (3,3),kshape2=(3,3))
    print(model.summary())
