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
        axes_scale = (1.0, scale, scale)
    else:
        axes_scale = (scale, scale, 1.0)

    return lambda x: skimage.transform.resize(x, numpy.multiply(x.shape, axes_scale), **kwargs)


def equalize(**kwargs):
    """
    Equalizes the image histogram, per channel.
    :param kwargs: Additional arguments for skimage.exposure.equalize_hist.
    :return: The equalize function.
    """
    def f(x):
        if keras.backend.image_data_format() == 'channels_last':
            x = numpy.moveaxis(x, -1, 0)

        y = numpy.empty_like(x, dtype=numpy.float64)

        for index, img in enumerate(x):
            y[index] = skimage.exposure.equalize_hist(img, **kwargs)

        if keras.backend.image_data_format() == 'channels_last':
            y = numpy.moveaxis(y, 0, -1)

        return y

    return f


def reduce_noise(**kwargs):
    """
    Reduces noise in the image.
    :param kwargs: Additional arguments for skimage.restoration.denoise_bilateral.
    :return: The reduce_noise function.
    """
    def f(x):
        if keras.backend.image_data_format() == 'channels_last':
            x = numpy.moveaxis(x, -1, 0)

        y = numpy.empty_like(x, dtype=numpy.float64)

        for index, img in enumerate(x):
            y[index] = skimage.restoration.denoise_bilateral(img, **kwargs)

        if keras.backend.image_data_format() == 'channels_last':
            y = numpy.moveaxis(y, 0, -1)

        return y

    return f
