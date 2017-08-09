import numpy
import skimage.data
import skimage.exposure
import skimage.restoration

import keras_microscopy.preprocessing.standardize


def test_desaturate_first(mocker):
    image_data_format = mocker.patch("keras.backend.image_data_format")
    image_data_format.return_value = "channels_first"
    img = numpy.random.rand(3, 100, 100)
    desaturate = keras_microscopy.preprocessing.standardize.desaturate(0.5)
    new = desaturate(img)
    x = img.mean()
    assert numpy.isclose(new.mean(), x + (img.max() - x) * 0.5)


def test_desaturate_last(mocker):
    image_data_format = mocker.patch("keras.backend.image_data_format")
    image_data_format.return_value = "channels_last"
    img = numpy.random.rand(100, 100, 3)
    desaturate = keras_microscopy.preprocessing.standardize.desaturate(0.5)
    new = desaturate(img)
    x = img.mean()
    assert numpy.isclose(new.mean(), x + (img.max() - x) * 0.5)


def test_rescale_first(mocker):
    image_data_format = mocker.patch("keras.backend.image_data_format")
    image_data_format.return_value = "channels_first"
    rescale = keras_microscopy.preprocessing.standardize.rescale(0.5, mode='reflect')
    img = numpy.random.rand(3, 100, 100)
    new = rescale(img)
    assert new.shape == (3, 50, 50)


def test_rescale_last(mocker):
    image_data_format = mocker.patch("keras.backend.image_data_format")
    image_data_format.return_value = "channels_last"
    rescale = keras_microscopy.preprocessing.standardize.rescale(0.5, mode='reflect')
    img = numpy.random.rand(100, 100, 3)
    new = rescale(img)
    assert new.shape == (50, 50, 3)


def test_equalize_first(mocker):
    image_data_format = mocker.patch("keras.backend.image_data_format")
    image_data_format.return_value = "channels_first"
    img = skimage.data.coffee()
    equalize = keras_microscopy.preprocessing.standardize.equalize()
    new = equalize(img)

    def f(x):
        y = numpy.empty_like(x, dtype=numpy.float64)
        for i in range(len(x[:, 0, 0])):
            y[i, :, :] = skimage.exposure.equalize_hist(x[i, :, :])
        return y
    numpy.testing.assert_array_equal(new, f(img))


def test_equalize_last(mocker):
    image_data_format = mocker.patch("keras.backend.image_data_format")
    image_data_format.return_value = "channels_last"
    img = numpy.moveaxis(skimage.data.coffee(), 2, 0)
    equalize = keras_microscopy.preprocessing.standardize.equalize()
    new = equalize(img)

    def f(x):
        y = numpy.empty_like(x, dtype=numpy.float64)
        for i in range(len(x[0, 0, :])):
            y[:, :, i] = skimage.exposure.equalize_hist(x[:, :, i])
        return y
    numpy.testing.assert_array_equal(new, f(img))


def test_reduce_noise_first(mocker):
    image_data_format = mocker.patch("keras.backend.image_data_format")
    image_data_format.return_value = "channels_first"
    img = numpy.random.rand(1, 100, 100)
    reduce = keras_microscopy.preprocessing.standardize.reduce_noise(multichannel=False)
    new = reduce(img)
    expected = skimage.restoration.denoise_bilateral(img[0, :, :], multichannel=False)
    numpy.testing.assert_array_equal(new, expected.reshape((1, 100, 100)))


def test_reduce_noise_last(mocker):
    image_data_format = mocker.patch("keras.backend.image_data_format")
    image_data_format.return_value = "channels_last"
    img = numpy.random.rand(100, 100, 1)
    reduce = keras_microscopy.preprocessing.standardize.reduce_noise(multichannel=False)
    new = reduce(img)
    expected = skimage.restoration.denoise_bilateral(img[:, :, 0], multichannel=False)
    numpy.testing.assert_array_equal(new, expected.reshape((100, 100, 1)))


def test_desaturate_uint8(mocker):
    image_data_format = mocker.patch("keras.backend.image_data_format")
    image_data_format.return_value = "channels_last"
    img = numpy.random.randint(256, size=(100, 100, 3)).astype(numpy.uint8)
    desaturate = keras_microscopy.preprocessing.standardize.desaturate(0.5)
    new = desaturate(img)
    x = img.mean()
    assert numpy.isclose(new.mean(), x + (img.max() - x) * 0.5)


def test_rescale_uint8(mocker):
    image_data_format = mocker.patch("keras.backend.image_data_format")
    image_data_format.return_value = "channels_first"
    rescale = keras_microscopy.preprocessing.standardize.rescale(0.5, mode='reflect')
    img = numpy.random.randint(256, size=(3, 100, 100)).astype(numpy.uint8)
    new = rescale(img)
    assert new.shape == (3, 50, 50)


def test_equalize_uint8(mocker):
    image_data_format = mocker.patch("keras.backend.image_data_format")
    image_data_format.return_value = "channels_first"
    img = numpy.random.randint(256, size=(3, 100, 100)).astype(numpy.uint8)
    equalize = keras_microscopy.preprocessing.standardize.equalize()
    new = equalize(img)

    def f(x):
        y = numpy.empty_like(x, dtype=numpy.float64)
        for i in range(len(x[:, 0, 0])):
            y[i, :, :] = skimage.exposure.equalize_hist(x[i, :, :])
        return y
    numpy.testing.assert_array_equal(new, f(img))


def test_reduce_noise_uint8(mocker):
    image_data_format = mocker.patch("keras.backend.image_data_format")
    image_data_format.return_value = "channels_first"
    img = numpy.random.randint(256, size=(1, 100, 100)).astype(numpy.uint8)
    reduce = keras_microscopy.preprocessing.standardize.reduce_noise(multichannel=False)
    new = reduce(img)
    expected = skimage.restoration.denoise_bilateral(img[0, :, :], multichannel=False)
    numpy.testing.assert_array_equal(new, expected.reshape((1, 100, 100)))
