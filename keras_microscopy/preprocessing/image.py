import keras_microscopy.preprocessing


class ImageGenerator(object):
    def flow_from_directory(
        self,
        directory,
        batch_size=32,
        seed=None,
        shape=(224, 224, 3),
        shuffle=True
    ):
        return keras_microscopy.preprocessing.DirectoryIterator(
            batch_size=batch_size,
            directory=directory,
            generator=self,
            seed=seed,
            shape=shape,
            shuffle=shuffle
        )
