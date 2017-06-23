import os.path

import numpy
import numpy.testing
import skimage.io

import keras_microscopy.preprocessing.image


class TestImageGenerator:
    def setup_class(self):
        image = numpy.random.random((64, 64, 5)).astype(numpy.float32)

        self.images = [image, image, image]

    def test_flow_from_directory(self, tmpdir):
        tmpdir.join("example").mkdir()

        count = 0

        for image in self.images:
            filename = "{}.tiff".format(count)

            pathname = os.path.join(tmpdir, "example", filename)

            skimage.io.imsave(pathname, image)

            count += 1

        generator = keras_microscopy.preprocessing.image.ImageGenerator()

        generator = generator.flow_from_directory(str(tmpdir), shape=(64, 64, 5))

        x, y = next(generator)

        numpy.testing.assert_equal(x.shape[0], y.shape[0])

        numpy.testing.assert_array_equal(x[0], self.images[0])
