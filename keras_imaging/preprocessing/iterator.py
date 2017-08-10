import concurrent.futures
import functools
import os
import threading

import keras.backend
import numpy
import six.moves
import skimage.io


def _count_filenames(directory, extensions, follow_links=False):
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

    samples = 0

    for root, _, files in _recursive_list(directory):
        for fname in files:
            is_valid = False

            for extension in extensions:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break

            if is_valid:
                samples += 1

    return samples


def _find_filenames(directory, extensions, class_indices, follow_links=False):
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

    classes = []

    filenames = []

    subdir = os.path.basename(directory)

    basedir = os.path.dirname(directory)

    for root, _, files in _recursive_list(directory):
        for fname in files:
            is_valid = False

            for extension in extensions:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break

            if is_valid:
                classes.append(class_indices[subdir])

                # add filename relative to directory
                absolute_path = os.path.join(root, fname)

                filenames.append(os.path.relpath(absolute_path, basedir))

    return classes, filenames


def _recursive_list(subpath):
    return sorted(os.walk(subpath), key=lambda tpl: tpl[0])


class Iterator(object):
    def __init__(self, n, batch_size, shuffle, seed):
        self.batch_index = 0

        self.batch_size = batch_size

        self.lock = threading.Lock()

        self.n = n

        self.shuffle = shuffle

        self.total_batches_seen = 0

        self.generator = self.sample(n, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def sample(self, n, batch_size=32, shuffle=False, seed=None):
        self.reset()

        while 1:
            if seed is not None:
                numpy.random.seed(seed + self.total_batches_seen)

            if self.batch_index == 0:
                index_array = numpy.arange(n)

                if shuffle:
                    index_array = numpy.random.permutation(n)

            index = (self.batch_index * batch_size) % n

            if n > index + batch_size:
                current_batch_size = batch_size

                self.batch_index += 1
            else:
                current_batch_size = n - index

                self.batch_index = 0

            self.total_batches_seen += 1

            indices = index_array[index: index + current_batch_size]

            yield (indices, index, current_batch_size)

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class DirectoryIterator(Iterator):
    def __init__(
            self,
            directory,
            generator,
            batch_size=32,
            sampling_method=None,
            seed=None,
            shape=(224, 224, 3),
            shuffle=True,
    ):
        self.directory = directory

        self.image_data_generator = generator

        self.sampling_method = sampling_method

        self.shape = shape

        extensions = {"png", "jpg", "jpeg", "bmp", "tiff", "tif"}

        self.samples = 0

        classes = []

        for subdirectory in sorted(os.listdir(directory)):
            if os.path.isdir(os.path.join(directory, subdirectory)):
                classes.append(subdirectory)

        self.num_class = len(classes)

        self.class_indices = dict(zip(classes, six.moves.range(len(classes))))

        func = functools.partial(_count_filenames, extensions=extensions)

        directories = [os.path.join(self.directory, subdirectory) for subdirectory in classes]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            self.samples = sum(executor.map(func, [directory for directory in directories]))

            self.filenames = []

            self.classes = numpy.zeros((self.samples,), dtype="int32")

            i = 0

            pathnames = [os.path.join(self.directory, subdirectory) for subdirectory in classes]

            futures = []

            for pathname in pathnames:
                future = executor.submit(_find_filenames, pathname, extensions, self.class_indices)

                futures.append(future)

            for future in futures:
                classes, filenames = future.result()

                self.classes[i:i + len(classes)] = classes

                self.filenames += filenames

                i += len(classes)

        if self.sampling_method:
            self.balance()

        super(DirectoryIterator, self).__init__(
            self.samples,
            batch_size,
            shuffle,
            seed
        )

    def balance(self):
        x = numpy.arange(self.samples).reshape((-1, 1))

        y = self.classes

        indices, self.classes = self.sampling_method.fit_sample(x, y)

        indices = indices.reshape(-1)

        self.filenames = [self.filenames[index] for index in indices]

        self.samples = self.classes.shape[0]

    def next(self):
        with self.lock:
            indicies, index, batch_size = next(self.generator)

        shape = (batch_size,) + self.shape

        batch_x = numpy.zeros(shape, dtype=keras.backend.floatx())

        for batch_index, index in enumerate(indicies):
            filename = os.path.join(self.directory, self.filenames[index])

            x = skimage.io.imread(filename)

            x = self.image_data_generator.standardize(x)

            x = self.image_data_generator.transform(x)

            batch_x[batch_index] = x

        shape = (len(batch_x), self.num_class)

        batch_y = numpy.zeros(shape, dtype=keras.backend.floatx())

        for batch_index, y in enumerate(self.classes[indicies]):
            batch_y[batch_index, y] = 1.0

        return batch_x, batch_y
