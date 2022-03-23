from typings.models import Model
from utils.dataset import Mnist as MnistDataset

import tensorflow as tf

keras = tf.keras


class Mnist(Model):
    def __init__(
        self,
        data_train: tf.data.Dataset,
        data_test: tf.data.Dataset,
        input_shape: tuple,
        optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(),
        loss: keras.losses.Loss = keras.losses.SparseCategoricalCrossentropy(),
        accuracy: keras.metrics.Accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        ),
        checkpoint_filepath: str = None,
    ):
        super().__init__(
            data_train,
            data_test,
            input_shape,
            optimizer,
            loss,
            accuracy,
            checkpoint_filepath,
        )

    def name(self) -> str:
        return "victim_mnist"

    def model(self) -> keras.Model:
        return keras.Sequential(
            [
                keras.layers.Conv2D(32, (3, 3), activation="relu"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(64, (3, 3), activation="relu"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(64, (3, 3), activation="relu"),
                keras.layers.Flatten(),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(10, activation="softmax"),
            ],
        )


if __name__ == "__main__":
    train_set, test_set = MnistDataset()
    mnist = Mnist(train_set, test_set, (28, 28, 1))
    mnist.train()
