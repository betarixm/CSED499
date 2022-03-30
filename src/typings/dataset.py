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
    def __init__(self, dataset):
        if not hasattr(dataset, "load_data"):
            raise TypeError

        self.ds = dataset

    @abstractmethod
    def postprocess(
        self, train: NumpyDataset, test: NumpyDataset
    ) -> Tuple[NumpyDataset, NumpyDataset]:
        pass

    def load_data(self) -> Tuple[NumpyDataset, NumpyDataset]:
        ds = self.ds

        (x_train, y_train), (x_test, y_test) = ds.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        (x_train, y_train), (x_test, y_test) = self.postprocess(
            (x_train, y_train), (x_test, y_test)
        )

        return (x_train, y_train), (x_test, y_test)

    def dataset(
        self, shuffle: int = 10000, batch: int = 32
    ) -> Tuple[TestSet, TrainSet]:
        (x_train, y_train), (x_test, y_test) = self.load_data()

        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(shuffle)
            .batch(batch)
        )

        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch)

        return train_ds, test_ds

    def __call__(self, *args, **kwargs):
        return self.dataset()


class ImageDataset(Dataset):
    def postprocess(
        self, train: NumpyDataset, test: NumpyDataset
    ) -> Tuple[NumpyDataset, NumpyDataset]:
        return train, test


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
