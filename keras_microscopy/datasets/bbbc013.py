# -*- coding: utf-8 -*-

import keras_microscopy.datasets


def load_data():
    return keras_microscopy.datasets.load_data("BBBC013")
