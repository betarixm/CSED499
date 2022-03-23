from typings.models import Model
from utils.dataset import Mnist as MnistDataset

import numpy as np
import tensorflow as tf

keras = tf.keras


class Mnist(Model):
    def __init__(
        self,
        data_train: tf.data.Dataset,
        data_test: tf.data.Dataset,
        input_shape: tuple,
        name: str = "victim_mnist",
        optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(),
        loss: keras.losses.Loss = keras.losses.SparseCategoricalCrossentropy(),
        accuracy: keras.metrics.Accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        ),
        checkpoint_filepath: str = None,
    ):
        super().__init__(
            name,
            data_train,
            data_test,
            input_shape,
            optimizer,
            loss,
            accuracy,
            checkpoint_filepath,
        )

    def _model(self) -> keras.Model:
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



    def pre_train(self):
        with self.tensorboard_file_writer().as_default():
            x = np.concatenate([x for x, y in self.data_test.take(1)], axis=0)
            tf.summary.image(f"{self.name()} test input", x, max_outputs=25, step=0)

    def post_train(self):
        pass

if __name__ == "__main__":
    train_set, test_set = MnistDataset()
    mnist = Mnist(train_set, test_set, (28, 28, 1))
    mnist.train()
