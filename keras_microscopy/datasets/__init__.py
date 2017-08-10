import os.path

import keras.utils.data_utils
import numpy


def load_data(name):
    hostname = "http://keras-imaging.storage.googleapis.com"

    origin = "{}/{}.tar.gz".format(hostname, name)

    pathname = keras.utils.data_utils.get_file(
        fname=name,
        origin=origin,
        untar=True
    )

    basename = "{}.npz".format(name)

    filename = os.path.join(pathname, basename)

    images, compounds, doses = numpy.load(filename)

    return images, (compounds, doses)
