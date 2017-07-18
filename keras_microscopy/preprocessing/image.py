import imblearn.over_sampling
import imblearn.under_sampling
import numpy
import skimage.filters
import skimage.transform
import skimage.util

import keras_microscopy.preprocessing


class ImageGenerator(object):
    def __init__(
        self,
        correct_distortion=False,
        correct_uneven_illumination=False,
        correct_vignetting=False,
        desaturate=None,
        equalize=None,
        flip_horizontally=False,
        flip_vertically=False,
        reduce_noise=None,
        remove_chromatic_aberration=False,
        rescale_intensity=None,
        rotate=False,
        smooth=False
    ):
        """
        :param correct_distortion:

        :param correct_uneven_illumination:

        :param correct_vignetting:

        :param desaturate:

        :param equalize:

        :param flip_horizontally:

        :param flip_vertically:

        :param reduce_noise:

        :param remove_chromatic_aberration:

        :param rescale_intensity:

        :param rotate:

        :param smooth:
        """
        self.correct_distortion = correct_distortion

        self.correct_uneven_illumination = correct_uneven_illumination

        self.correct_vignetting = correct_vignetting

        self.desaturate = desaturate

        self.equalize = equalize

        self.flip_horizontally = flip_horizontally

        self.flip_vertically = flip_vertically

        self.reduce_noise = reduce_noise

        self.remove_chromatic_aberration = remove_chromatic_aberration

        self.rescale = rescale_intensity

        self.rotate = rotate

        self.smooth = smooth

    def flow_from_directory(
        self,
        directory,
        batch_size=32,
        sampling_method=None,
        seed=None,
        shape=(224, 224, 3),
        shuffle=True
    ):
        """
        :param directory:

        :param batch_size: int

        :param sampling_method:

        :param seed: int or None; optional

        :param shape: tuple of ints

        :param shuffle: boolean; optional

        :return:

        :rtype: keras_microscopy.preprocessing.DirectoryIterator
        """
        if sampling_method == "oversample":
            sampling_method = imblearn.over_sampling.RandomOverSampler(random_state=seed)
        elif sampling_method == "undersample":
            sampling_method = imblearn.under_sampling.RandomUnderSampler(random_state=seed)

        return keras_microscopy.preprocessing.DirectoryIterator(
            batch_size=batch_size,
            directory=directory,
            generator=self,
            sampling_method=sampling_method,
            seed=seed,
            shape=shape,
            shuffle=shuffle
        )

    def standardize(self, x):
        """
        :param x:

        :return:
        """
        if self.desaturate:
            x = self.desaturate(x)

        if self.rescale:
            x = self.rescale(x)

        if self.equalize:
            x = self.equalize(x)

        if self.reduce_noise:
            x = self.reduce_noise(x)

        return x

    def transform(self, x):
        """
        :param x:

        :return:
        """
        if self.flip_horizontally:
            if numpy.random.random() < 0.5:
                x = numpy.fliplr(x)

        if self.flip_vertically:
            if numpy.random.random() < 0.5:
                x = numpy.flipud(x)

        if self.rotate:
            k = numpy.pi / 360 * numpy.random.uniform(-360, 360)

            x = skimage.transform.rotate(x, k)

        if self.smooth:
            x = skimage.filters.gaussian(x, numpy.random.random())

        return x
