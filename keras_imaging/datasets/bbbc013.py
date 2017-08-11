# -*- coding: utf-8 -*-

import keras_imaging.datasets


def load_data():
    """
    Load Ravkin, et al.’s translocation dataset.

    Ravkin, et al.’s translocation dataset is a collection of images of
    cytoplasm-nucleus translocation of the Forkhead fusion protein (FKHR-EGFP)
    in stably transfected human osteosarcoma cells (U2OS). Images are
    accompanied by each well’s treatment compound and concentration.

    :return: A three-tuple of images, compounds, and concentrations.

    Images are 640 × 640 × 2. The first channel of each image was stained for
    FKHR-GFP and the second channel of each image was stained for DNA.

    A compound is either LY294002 (“1”) or Wortmannin (“2”). If a compound is
    “0”, it’s either a negative control, positive control, or untreated.

    A concentration is the concentration of the aforementioned compound.

    :rtype: ndarray, (ndarray, ndarray)
    """
    return keras_imaging.datasets.load_data("BBBC013")
