import glob
import numpy as np
import argparse
import h5py
import tensorflow as tf
from matplotlib import pyplot as plt



def make_complex_k(ksapce_real_imaginary):
    real = ksapce_real_imaginary[:, :, :, 0]
    imag = ksapce_real_imaginary[:, :, :, 1]
    k_complex = tf.complex(real, imag)
    return k_complex

def ifft_shift_layer(kspace):
    reconstruct_image = (tf.abs(tf.signal.fftshift(tf.signal.ifft2d(kspace))))
    return reconstruct_image


def kspace_image_display(full_k, under_k):
    for slice_position in range(full_k.shape[0]):
        full_ks = tf.math.log(tf.abs(full_k[slice_position]))
        plt.subplot(221),plt.imshow(full_ks,cmap = 'gray' )
        plt.title('Ground Truth--k'), plt.xticks([]), plt.yticks([])

        full_i = ifft_shift_layer(full_k)
        plt.subplot(222),plt.imshow((full_i[slice_position]), cmap = 'gray' )
        plt.title('Ground Truth--i'), plt.xticks([]), plt.yticks([])

        under_ks = tf.math.log(tf.abs(under_k[slice_position]))
        plt.subplot(223),plt.imshow(under_ks, cmap = 'gray' )
        plt.title('under-k'), plt.xticks([]), plt.yticks([])

        under_i = ifft_shift_layer(under_k)
        plt.subplot(224),plt.imshow((under_i[slice_position]),cmap = 'gray')
        plt.title('under-i'), plt.xticks([]), plt.yticks([])
        if (slice_position >= 10):
            break
        print(slice_position)
        plt.show()

def kspace_display(full_k, under_k):
    for slice_position in range(full_k.shape[0]):
        full_ks = tf.math.log(tf.abs(full_k[slice_position]))
        plt.subplot(121),plt.imshow(full_ks[100], cmap = 'gray' )
        plt.title(''), plt.xticks([]), plt.yticks([])

        under_ks = tf.math.log(tf.abs(under_k[slice_position]))
        plt.subplot(122),plt.imshow(under_ks[100],cmap = 'gray')
        plt.title(''), plt.xticks([]), plt.yticks([])

        '''if (slice_position >= 10):
            break'''
        plt.show()

def image_display(image_f, image_u):
    for slice_position in range(image_f.shape[0]):
        plt.subplot(121), plt.imshow((image_f[slice_position]), cmap = 'gray' )
        plt.title('Ground Truth'), plt.xticks([]), plt.yticks([])

        plt.subplot(122), plt.imshow((image_u[slice_position]), cmap = 'gray')
        plt.title('Under sampled', plt.xticks([]), plt.yticks([]))
        if (slice_position >= 10):
            break
        plt.show()



def image_save(full_image, under_image):
    slice_position = 0
    for i in range((full_image.shape[0])):
        image = (full_image[i])
        #plt.imsave('E:/Code/Python/MRI_de_aliasing/data/FastMRI_T1/reconstruct/original/{}.jpg'.format(slice_position), image, cmap = 'gray')

        image2 = (under_image[i])
        plt.imsave('E:/Code/RA_Unet/output/input_af_6/{}.jpg'.format(slice_position), image2, cmap = 'gray')
        slice_position = slice_position+1
    print('successfully saved {} images into {}'.format(under_image.shape[0]))

#E:\Code\RA_Unet\output\input_af_6
under_rate = '06'

stats = np.load('../data/stats_fs_unet_norm_' + '20'+ '.npy')

var_sampling_mask = np.load("../data/sampling_mask_" + under_rate + "perc.npy")
var_sampling_mask = np.fft.ifftshift(var_sampling_mask)
print("Undersampling value:", 1.0*var_sampling_mask.sum()/var_sampling_mask.size)
print("Mask type:",  var_sampling_mask.dtype)
plt.figure()
plt.imshow(var_sampling_mask,cmap = "gray")
plt.axis("off")
#plt.show()

orig = np.load("../data/test/1.npy")
print('original k shape', orig.shape)
imshape = (256, 256)
norm = np.sqrt(imshape[0] * imshape[1])
kspace_full = orig/norm

kspace_under = kspace_full.copy()
kspace_under[:, var_sampling_mask, :] = 0


#make complex data
complex_f_k = (make_complex_k(kspace_full))
kspace_full= 0
complex_u_k = (make_complex_k(kspace_under))
kspace_under = 0
#kspace_display(complex_f_k, complex_u_k)

full_ks = tf.math.log(tf.abs(complex_f_k))
under_ks = tf.math.log(tf.abs(complex_u_k))

slice = 100
plt.figure()
plt.subplot(121),plt.imshow(full_ks[slice],)
plt.title(''), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(under_ks[slice])
plt.title(''), plt.xticks([]), plt.yticks([])
#plt.show()


#generate image
complex_f_k = ifft_shift_layer(complex_f_k)
print('combine shape', complex_f_k.shape)

complex_u_k = ifft_shift_layer(complex_u_k)
print('combine u k shape', complex_u_k.shape)

#image_display(complex_f_k, complex_u_k)
image_save(complex_f_k, complex_u_k)
#kspace_image_display(complex_f_k, complex_u_k)'''

