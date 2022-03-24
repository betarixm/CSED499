from typing import Tuple
from typings.dataset import ImageDataset, GrayscaleMixin, NoisyMixin
from typings.dataset import NumpyDataset

import tensorflow as tf

keras = tf.keras


class Mnist(ImageDataset, GrayscaleMixin):
    def __init__(self):
        super().__init__(keras.datasets.mnist)

    def postprocess(
        self, train: NumpyDataset, test: NumpyDataset
    ) -> Tuple[NumpyDataset, NumpyDataset]:
        (x_train, y_train), (x_test, y_test) = GrayscaleMixin.process(train, test)
        return (x_train, y_train), (x_test, y_test)


class NoisyMnist(ImageDataset, GrayscaleMixin, NoisyMixin):
    def __init__(self):
        super().__init__(keras.datasets.mnist)

    def postprocess(
        self, train: NumpyDataset, test: NumpyDataset
    ) -> Tuple[NumpyDataset, NumpyDataset]:
        train, test = GrayscaleMixin.process(train, test)
        (x_train, y_train), (x_test, y_test) = NoisyMixin.process(train, test)
        return (x_train, y_train), (x_test, y_test)


        x_train_noisy, x_test_noisy = noisy(x_train), noisy(x_test)

        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train_noisy, x_train))
            .shuffle(10000)
            .batch(32)
        )

        test_ds = tf.data.Dataset.from_tensor_slices((x_test_noisy, x_test)).batch(32)

        return train_ds, test_ds
