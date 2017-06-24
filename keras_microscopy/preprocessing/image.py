import keras_microscopy.preprocessing


class ImageGenerator(object):
    def flow_from_directory(
        self,
        directory,
        batch_size=32,
        sampling_method=None,
        seed=None,
        shape=(224, 224, 3),
        shuffle=True
    ):
        return keras_microscopy.preprocessing.DirectoryIterator(
            batch_size=batch_size,
            directory=directory,
            generator=self,
            sampling_method=sampling_method,
            seed=seed,
            shape=shape,
            shuffle=shuffle
        )

    @staticmethod
    def standardize(x):
        return x

    @staticmethod
    def transform(x):
        return x
