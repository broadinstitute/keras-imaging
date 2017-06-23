import itertools
import os.path

import imblearn.over_sampling
import imblearn.under_sampling
import numpy
import numpy.testing
import skimage.io

import keras_microscopy.preprocessing.image


class TestImageGenerator:
    def setup_class(self):
        images = []

        images.extend(itertools.repeat(numpy.random.random((64, 64, 5)), 100))

        images = [image.astype(numpy.float32) for image in images]

        self.images = images

    def test_flow_from_directory(self, tmpdir):
        for directory in ["a", "b", "c"]:
            tmpdir.join(directory).mkdir()

        count = 0

        for image in self.images:
            filename = "{}.tiff".format(count)

            directory = numpy.random.choice(["a", "b", "c"])

            pathname = os.path.join(tmpdir, directory, filename)

            skimage.io.imsave(pathname, image)

            count += 1

        generator = keras_microscopy.preprocessing.image.ImageGenerator()

        generator = generator.flow_from_directory(str(tmpdir), shape=(64, 64, 5))

        x, y = next(generator)

        numpy.testing.assert_equal(x.shape[0], y.shape[0])

        numpy.testing.assert_array_equal(x[0], self.images[0])

    def test_sampling_method(self, tmpdir):
        for directory in ["a", "b", "c"]:
            tmpdir.join(directory).mkdir()

        count = 0

        for image in self.images:
            filename = "{}.tiff".format(count)

            directory = numpy.random.choice(["a", "b", "c"])

            pathname = os.path.join(tmpdir, directory, filename)

            skimage.io.imsave(pathname, image)

            count += 1

        generator = keras_microscopy.preprocessing.image.ImageGenerator()

        sampling_method = imblearn.over_sampling.RandomOverSampler()

        generator = generator.flow_from_directory(
            directory=str(tmpdir),
            sampling_method=sampling_method,
            shape=(64, 64, 5)
        )

        a = generator.classes[generator.classes == 0].shape
        b = generator.classes[generator.classes == 1].shape
        c = generator.classes[generator.classes == 2].shape

        assert a == b == c

        assert generator.n == generator.classes.shape[0]

        assert generator.samples == generator.classes.shape[0]

        generator = keras_microscopy.preprocessing.image.ImageGenerator()

        sampling_method = imblearn.under_sampling.RandomUnderSampler()

        generator = generator.flow_from_directory(
            directory=str(tmpdir),
            sampling_method=sampling_method,
            shape=(64, 64, 5)
        )

        a = generator.classes[generator.classes == 0].shape
        b = generator.classes[generator.classes == 1].shape
        c = generator.classes[generator.classes == 2].shape

        assert a == b == c

        assert generator.n == generator.classes.shape[0]

        assert generator.samples == generator.classes.shape[0]
