import pytest

import keras_microscopy.preprocessing.image


@pytest.fixture
def image_generator():
    return keras_microscopy.preprocessing.image.ImageGenerator()
