import matplotlib.pylab as plt
import numpy as np

def gaussian1d(pattern_shape, factor,direction = "column",center=None, cov=None):
    """
    Description: creates a 1D gaussian sampling pattern either in the row or column direction
    of a 2D image
    :param factor: sampling factor in the desired direction
    :param direction: sampling direction, 'row' or 'column'
    :param pattern_shape: shape of the desired sampling pattern.
    :param center: coordinates of the center of the Gaussian distribution
    :param cov: covariance matrix of the distribution
    :return: sampling pattern image. It is a boolean image
    """

    if direction != "column":
        pattern_shape = (pattern_shape[1],pattern_shape[0])

    if center is None:
        center = np.array([1.0 * pattern_shape[1] / 2 - 0.5])

    if cov is None:
        cov = np.array([[(1.0 * pattern_shape[1] / 4) ** 2]])


    factor = int(factor * pattern_shape[1])

    samples = np.array([0])

    m = 1  # Multiplier. We have to increase this value
    # until the number of points (disregarding repeated points)
    # is equal to factor

    while (samples.shape[0] < factor):

        samples = np.random.multivariate_normal(center, cov, m * factor)
        samples = np.rint(samples).astype(int)
        indexes = np.logical_and(samples >= 0, samples < pattern_shape[1])
        samples = samples[indexes]
        samples = np.unique(samples)
        if samples.shape[0] < factor:
            m *= 2
            continue

    indexes = np.arange(samples.shape[0], dtype=int)
    np.random.shuffle(indexes)
    samples = samples[indexes][:factor]

    under_pattern = np.zeros(pattern_shape, dtype=bool)
    under_pattern[:, samples] = True

    if direction != "column":
        under_pattern = under_pattern.T

    return under_pattern


def gaussian2d(pattern_shape, factor, center=None, cov=None):
    """
    Description: creates a 2D gaussian sampling pattern of a 2D image
    :param factor: sampling factor in the desired direction
    :param pattern_shape: shape of the desired sampling pattern.
    :param center: coordinates of the center of the Gaussian distribution
    :param cov: covariance matrix of the distribution
    :return: sampling pattern image. It is a boolean image
    """
    N = pattern_shape[0] * pattern_shape[1]  # Image length

    factor = int(N * factor)

    if center is None:
        center = np.array([1.0 * pattern_shape[0] / 2 - 0.5, 1.0 * pattern_shape[1] / 2 - 0.5])

    if cov is None:
        cov = np.array([[(1.0 * pattern_shape[0] / 4) ** 2, 0], [0, (1.0 * pattern_shape[1] / 4) ** 2]])

    samples = np.array([0])

    m = 1  # Multiplier. We have to increase this value
    # until the number of points (disregarding repeated points)
    # is equal to factor

    while (samples.shape[0] < factor):
        samples = np.random.multivariate_normal(center, cov, m * factor)
        samples = np.rint(samples).astype(int)
        indexesx = np.logical_and(samples[:, 0] >= 0, samples[:, 0] < pattern_shape[0])
        indexesy = np.logical_and(samples[:, 1] >= 0, samples[:, 1] < pattern_shape[1])
        indexes = np.logical_and(indexesx, indexesy)
        samples = samples[indexes]
        # samples[:,0] = np.clip(samples[:,0],0,input_shape[0]-1)
        # samples[:,1] = np.clip(samples[:,1],0,input_shape[1]-1)
        samples = np.unique(samples[:, 0] + 1j * samples[:, 1])
        samples = np.column_stack((samples.real, samples.imag)).astype(int)
        if samples.shape[0] < factor:
            m *= 2
            continue

    indexes = np.arange(samples.shape[0], dtype=int)
    np.random.shuffle(indexes)
    samples = samples[indexes][:factor]

    under_pattern = np.zeros(pattern_shape, dtype=bool)
    under_pattern[samples[:, 0], samples[:, 1]] = True
    return under_pattern


pattern_shape = (256,256)
factor= 0.84
af= 6
gussian_mask_uniform = gaussian1d(pattern_shape, factor, direction = "row")
gussian_mask_random = gaussian2d(pattern_shape,factor= factor)
#np.save('../mask/gaussian_mask_AF-{50}_2D_256X256.npy',gussian_mask)

gussian_mask = gussian_mask_random
uncentered_mask= np.fft.fftshift(gussian_mask)

plt.figure()
plt.subplot(121)
plt.imshow(gussian_mask, cmap='gray')
plt.axis("off")
plt.title("center mask of AF={}".format(af))

plt.subplot(122)
plt.imshow(uncentered_mask, cmap='gray')
plt.axis("off")
plt.title("uncentered sampling mask of AF={}".format(af))
plt.show()
#np.save('../data/sampling_mask_06perc.npy',gussian_mask)