import matplotlib.pylab as plt
import numpy as np
under_rate = '06'
imshape = (256,256)
norm = np.sqrt(imshape[0]*imshape[1])
nchannels = 2 #complex data real + imag

# undersampling patterns - uncentred k-space
var_sampling_mask = np.load("../data/sampling_mask_" + under_rate + "perc.npy")

print("Undersampling value:", 1.0*var_sampling_mask.sum()/var_sampling_mask.size)
print("Mask type:",  var_sampling_mask.dtype)
plt.figure()
plt.imshow(var_sampling_mask,cmap = "gray")
plt.axis("off")
plt.show()