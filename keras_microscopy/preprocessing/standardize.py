import keras
import numpy
import skimage.exposure
import skimage.restoration
import skimage.transform


def desaturate(ratio):
    """
    Fades the image to white. Higher ratio means more fading.
    :param ratio: The ratio to fade by, between 0 and 1.
    :return: The desaturate function.
    """
    return lambda x: x + (x.max() - x) * ratio


def rescale(scale, **kwargs):
    """
    Rescales the image according to the scale ratio.
    :param scale: The scalar to rescale the image by.
    :param kwargs: Additional arguments for skimage.transform.resize.
    :return: The rescale function.
    """
    if keras.backend.image_data_format() == 'channels_first':
        return lambda x: skimage.transform.resize(x, (len(x[:, 0, 0]), len(x[0, :, 0]) * scale, len(x[0, 0, :]) * scale), **kwargs)
    else:
        return lambda x: skimage.transform.resize(x, (len(x[:, 0, 0]) * scale, len(x[0, :, 0]) * scale, len(x[0, 0, :])), **kwargs)


def equalize(**kwargs):
    """
    Equalizes the image histogram, per channel.
    :param kwargs: Additional arguments for skimage.exposure.equalize_hist.
    :return: The equalize function.
    """
    if keras.backend.image_data_format() == 'channels_first':
        def f(x):
            y = numpy.empty_like(x, dtype=numpy.float64)
            for i in range(len(x[:, 0, 0])):
                y[i, :, :] = skimage.exposure.equalize_hist(x[i, :, :], **kwargs)
            return y
        return f
    else:
        def f(x):
            y = numpy.empty_like(x, dtype=numpy.float64)
            for i in range(len(x[0, 0, :])):
                y[:, :, i] = skimage.exposure.equalize_hist(x[:, :, i], **kwargs)
            return y
        return f


def reduce_noise(**kwargs):
    """
    Reduces noise in the image.
    :param kwargs: Additional arguments for skimage.restoration.denoise_bilateral.
    :return: The reduce_noise function.
    """
    if keras.backend.image_data_format() == 'channels_first':
        def f(x):
            y = numpy.empty_like(x, dtype=numpy.float64)
            for i in range(len(x[:, 0, 0])):
                y[i, :, :] = skimage.restoration.denoise_bilateral(x[i, :, :], **kwargs)
            return y
        return f
    else:
        def f(x):
            y = numpy.empty_like(x, dtype=numpy.float64)
            for i in range(len(x[0, 0, :])):
                y[:, :, i] = skimage.restoration.denoise_bilateral(x[:, :, i], **kwargs)
            return y
        return f
