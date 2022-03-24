from abc import ABC, abstractmethod
from typing import NewType, Tuple, Protocol

import numpy as np
import tensorflow as tf

keras = tf.keras

TestSet = NewType("TestSet", tf.data.Dataset)
TrainSet = NewType("TrainSet", tf.data.Dataset)

NumpyDataset = Tuple[np.array, np.array]


class DatasetPostprocessMixin(Protocol):
    @staticmethod
    def process(
        train: NumpyDataset, test: NumpyDataset
    ) -> Tuple[NumpyDataset, NumpyDataset]:
        ...


class Dataset(ABC):
    @staticmethod
    @abstractmethod
    def dataset() -> (TestSet, TrainSet):
        pass



class GrayscaleMixin(DatasetPostprocessMixin):
    @staticmethod
    def process(
        train: NumpyDataset, test: NumpyDataset
    ) -> Tuple[NumpyDataset, NumpyDataset]:
        (x_train, y_train), (x_test, y_test) = train, test

        x_train = x_train[..., tf.newaxis].astype("float32")
        x_test = x_test[..., tf.newaxis].astype("float32")

        return (x_train, y_train), (x_test, y_test)


class NoisyMixin(DatasetPostprocessMixin):
    @staticmethod
    def process(
        train: NumpyDataset, test: NumpyDataset
    ) -> Tuple[NumpyDataset, NumpyDataset]:
        def noisy(ds: np.array):
            noise = 0.1 * np.random.normal(size=np.shape(ds))
            return np.clip(ds + noise, 0.0, 1.0)

        (x_train, y_train), (x_test, y_test) = train, test

        return (noisy(x_train), x_train), (noisy(x_test), x_test)
