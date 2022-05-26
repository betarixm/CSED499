from typing import Tuple
from typings.dataset import (
    Datasets,
    ImageDataset,
    GrayscaleMixin,
    NoisyMixin,
    SlqMixin,
    IdemMixin,
)
from typings.dataset import NumpyDataset

import tensorflow as tf
import numpy as np

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


class Cifar10(ImageDataset):
    def __init__(self):
        super().__init__(keras.datasets.cifar10)


class NoisyCifar10(ImageDataset, NoisyMixin):
    def __init__(self):
        super().__init__(keras.datasets.cifar10)

    def postprocess(
        self, train: NumpyDataset, test: NumpyDataset
    ) -> Tuple[NumpyDataset, NumpyDataset]:
        (x_train, y_train), (x_test, y_test) = NoisyMixin.process(train, test)
        return (x_train, y_train), (x_test, y_test)


class SlqCifar10(ImageDataset, SlqMixin):
    def __init__(self):
        super().__init__(keras.datasets.cifar10)

    def postprocess(
        self, train: NumpyDataset, test: NumpyDataset
    ) -> Tuple[NumpyDataset, NumpyDataset]:
        (x_train, y_train), (x_test, y_test) = SlqMixin.process(train, test)
        return (x_train, y_train), (x_test, y_test)


class NoisySlqCifar10(ImageDataset, SlqMixin):
    def __init__(self):
        super().__init__(keras.datasets.cifar10)

    def postprocess(
        self, train: NumpyDataset, test: NumpyDataset
    ) -> Tuple[NumpyDataset, NumpyDataset]:
        def noisy(ds: np.array):
            noise = 0.1 * np.random.normal(size=np.shape(ds))
            return np.clip(ds + noise, 0.0, 1.0)

        (x_train, y_train), (x_test, y_test) = train, test

        (x_train, y_train), (x_test, y_test) = SlqMixin.process(
            (noisy(x_train), y_train), (noisy(x_test), y_test)
        )

        return (x_train, y_train), (x_test, y_test)


class IdemCifar10(ImageDataset, IdemMixin):
    def __init__(self):
        super().__init__(keras.datasets.cifar10)

    def postprocess(
        self, train: NumpyDataset, test: NumpyDataset
    ) -> Tuple[NumpyDataset, NumpyDataset]:
        (x_train, y_train), (x_test, y_test) = IdemMixin.process(train, test)
        return (x_train, y_train), (x_test, y_test)


class ExCifar10(Datasets):
    def __init__(self):
        super().__init__([IdemCifar10(), NoisyCifar10(), SlqCifar10()])
