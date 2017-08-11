import keras_imaging.datasets.bbbc013


def test_load_data():
    images, (compounds, doses) = keras_imaging.datasets.bbbc013.load_data()

    assert images.shape == (100, 640, 640, 2)

    assert compounds.shape == (100,)

    assert doses.shape == (100,)
