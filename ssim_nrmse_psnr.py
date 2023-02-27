import matplotlib.pylab as plt
import numpy as np
import os
import skimage.measure as meas
import natsort     #Natural sorting for python
import glob
from skimage.metrics import normalized_root_mse as nrmse, structural_similarity as ssim, peak_signal_noise_ratio as psnr, mean_squared_error as mse
from sewar import full_ref
from sklearn.metrics import mean_squared_error
import math

def PSNR(original, predictions):
    mse = np.mean((original - predictions) ** 2)
    if mse == 0:
        return 100
    pixel_max = 255.0
    Psnr = 15 * math.log10((pixel_max / math.sqrt(mse)))
    return Psnr

def NRMSE(original, prediction):
    rms = math.sqrt(mean_squared_error(original, prediction))
    Nrmse = rms/(original.max() - prediction.min())
    return Nrmse

under_rate = '06'
under_rate2 = '20'
var_sampling_mask = np.load("../data/sampling_mask_" + under_rate + "perc.npy")
#var_sampling_mask = np.fft.ifftshift(var_sampling_mask)
stats = np.load('../data/stats_fs_unet_norm_' + under_rate2+ '.npy')

orig = np.load("../data/test/1.npy")
print('original k shape', orig.shape)
imshape = (256, 256)
norm = np.sqrt(imshape[0] * imshape[1])
kspace_full = orig/norm
rec = np.abs(np.fft.ifft2(kspace_full[:, :, :, 0] + 1j * kspace_full[:, :, :, 1])).astype(np.float64)
print('original image shape', rec.shape)
kspace_under = kspace_full.copy()
kspace_full = 0
kspace_under[:, var_sampling_mask, :] = 0
kspace_under = (kspace_under - stats[0]) / stats[1]
print('original uk shape', kspace_under.shape)
under_rec = np.abs(np.fft.ifft2(kspace_under[:, :, :, 0] + 1j * kspace_under[:, :, :, 1])).astype(np.float64)
print('under_rec image shape', under_rec.shape)



from tensorflow.keras.optimizers import Adam
#import networks as fsnet
import wnet_network as fsnet
model = fsnet.wnet(stats[0], stats[1], stats[2], stats[3])
opt = Adam(lr=1e-3, decay=1e-7)
model.compile(loss=[fsnet.nrmse, fsnet.nrmse], optimizer=opt, loss_weights=[0.01, 0.99])
model_name = "./MD_20/2022-09-23_1642_1501/rdau.h5"
model.load_weights(model_name)
pred = model.predict(kspace_under)[1].astype(np.float64)
print('predict image shape', pred.shape)
pred = pred.reshape((rec.shape[0], rec.shape[1], rec.shape[2]))
print('prediction image shape after reshape', pred.shape)

met = np.zeros([3, rec.shape[0]])

for slice_position in range(rec.shape[0]): 
    ref_img = rec[slice_position]
    noisy_test1 = under_rec[slice_position]
    restore_image = pred[slice_position]

    ssim_score = ssim(ref_img.ravel(), restore_image.ravel(), win_size = ref_img.size-1)
    psnr_skimg = PSNR(ref_img, restore_image)
    rmse_skimg = NRMSE(ref_img, restore_image)

    #(ref[ii].ravel(), hyb[ii].ravel(), win_size = ref[slice_position].size-1)
    #print("\nslice position: %.0f, ssim: %.2f, psnr: %.2f, nrmse: %.2f" % (slice_position, ssim_score, psnr_skimg, rmse_skimg))

    met[0, slice_position] = ssim_score
    met[1, slice_position] = psnr_skimg
    met[2, slice_position] = rmse_skimg
    #slice_position = slice_position + 1

    #np.save('E:/BIDDUT/MD_RDA_UNET/output/ra_5_reconstruct/met_{}.npy'.format(under_rate), met)

    # display
    plt.subplot(131), plt.imshow(ref_img, )
    plt.title('Fully Sampled '), plt.xticks([]), plt.yticks([])
    #plt.imsave('E:/BIDDUT/MD_RDA_UNET/output/original/{}.jpg'.format(slice_position), ref_img, cmap='gray')
    plt.subplot(132), plt.imshow(noisy_test1, )
    plt.title("Under Sampled ")
    #plt.imsave('E:/BIDDUT/MD_RDA_UNET/output/input_af_5/{}.jpg'.format(slice_position), noisy_test1, cmap='gray')
    plt.subplot(133), plt.imshow(restore_image, )
    plt.title('Reconstruct image')
    #plt.imsave('E:/BIDDUT/MD_RDA_UNET/output/ra_5_reconstruct/{}.jpg'.format(slice_position), restore_image, cmap='gray')
    #plt.show()
     # if slice_position>= 10:
     #     break

dataset = 'calgary_T1'

print('\nModel RDUA: {}'.format(model_name))
print('under sampled: ', under_rate)
print('Input : {}'.format(dataset))

print("\nResult: mean+/-std ")
print("ssim: %.3f +/- %.2f" % ((met[0, :].mean()), met[0, :].std()))
print("psnr: %.2f +/- %.2f" % (met[1, :].mean(), met[1, :].std()))
print("nrmse: %.3f +/- %.2f" % (met[2, :].mean(), met[2, :].std()))

print("\nmaximum ssim: %.3f and slice number %.0f" % (np.max(met[0]), np.argmax(met[0])))
print("maximum  psnr : %.2f and slice number %.0f" % (np.max(met[1]), np.argmax(met[1])))
print("minimum  nrmse: %.3f and slice number %.0f" % (np.min(met[2]), np.argmin(met[2])))

print('successful')
