from abc import ABC
from typing import NewType, Tuple, Protocol, List

import numpy as np
import tensorflow as tf

from defense.models import Denoiser

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
        if dataset is not None and not hasattr(dataset, "load_data"):
            raise TypeError

        self.ds = dataset

    def postprocess(
        self, train: NumpyDataset, test: NumpyDataset
    ) -> Tuple[NumpyDataset, NumpyDataset]:
        return train, test

    def load_data(self) -> Tuple[NumpyDataset, NumpyDataset]:
        ds = self.ds

        (x_train, y_train), (x_test, y_test) = ds.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        (x_train, y_train), (x_test, y_test) = self.postprocess(
            (x_train, y_train), (x_test, y_test)
        )

        return (x_train, y_train), (x_test, y_test)

    def dataset(
        self, shuffle: int = 262144, batch: int = 32
    ) -> Tuple[TestSet, TrainSet]:
        (x_train, y_train), (x_test, y_test) = self.load_data()

        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(shuffle)
            .batch(batch)
        )

        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch)

        return train_ds, test_ds

    def __call__(self, *args, **kwargs):
        return self.dataset()


class ImageDataset(Dataset):
    pass


class Datasets(Dataset, ABC):
    def __init__(self, datasets: List[Dataset]):
        super().__init__(None)
        self.datasets: List[Dataset] = datasets

    def load_data(self) -> Tuple[NumpyDataset, NumpyDataset]:
        datasets_stack = [
            (_x_train, _y_train, _x_test, _y_test)
            for (_x_train, _y_train), (_x_test, _y_test) in [
                ds.load_data() for ds in self.datasets
            ]
        ]

        categorical_stack = map(list, zip(*datasets_stack))

        x_train, y_train, x_test, y_test = map(np.concatenate, categorical_stack)

        (x_train, y_train), (x_test, y_test) = self.postprocess(
            (x_train, y_train), (x_test, y_test)
        )

        return (x_train, y_train), (x_test, y_test)


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


class SlqMixin(DatasetPostprocessMixin):
    @staticmethod
    def process(
        train: NumpyDataset, test: NumpyDataset
    ) -> Tuple[NumpyDataset, NumpyDataset]:
        (x_train, y_train), (x_test, y_test) = train, test

        defense_model = Denoiser(
            f"denoiser_dataset_generator", input_shape=x_train.shape[-3:]
        )

        def noisy(ds: np.array):
            return defense_model.predict(ds)

        return (noisy(x_train), x_train), (noisy(x_test), x_test)


class IdemMixin(DatasetPostprocessMixin):
    @staticmethod
    def process(
        train: NumpyDataset, test: NumpyDataset
    ) -> Tuple[NumpyDataset, NumpyDataset]:
        (x_train, _), (x_test, _) = train, test

        return (x_train, x_train), (x_test, x_test)
