from typings.dataset import Dataset, TestSet, TrainSet

import numpy as np
import tensorflow as tf

keras = tf.keras


class Mnist(Dataset):
    @staticmethod
    def load_data():
        mnist = keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        x_train = x_train[..., tf.newaxis].astype("float32")
        x_test = x_test[..., tf.newaxis].astype("float32")

        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def dataset() -> (TestSet, TrainSet):
        (x_train, y_train), (x_test, y_test) = Mnist.load_data()

        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(10000)
            .batch(32)
        )

        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

        return train_ds, test_ds


class NoisyMnist(Dataset):
    @staticmethod
    def dataset() -> (TestSet, TrainSet):
        def noisy(ds):
            noise = 0.1 * np.random.normal(size=np.shape(ds))
            return np.clip(ds + noise, 0.0, 1.0)

        (x_train, y_train), (x_test, y_test) = Mnist.load_data()

        x_train_noisy, x_test_noisy = noisy(x_train), noisy(x_test)

        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train_noisy, x_train))
            .shuffle(10000)
            .batch(32)
        )

        test_ds = tf.data.Dataset.from_tensor_slices((x_test_noisy, x_test)).batch(32)

        return train_ds, test_ds
