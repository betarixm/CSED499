from typing import List
from typings.models import Model
from utils.dataset import NoisyMnist

import numpy as np
import tensorflow as tf

keras = tf.keras


class Reformer(Model):
    def __init__(
        self,
        data_train: tf.data.Dataset,
        data_test: tf.data.Dataset,
        input_shape: tuple,
        name: str = "defense_reformer",
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
        def layer_conv2d():
            return keras.layers.Conv2D(
                3,
                (3, 3),
                activation="sigmoid",
                padding="same",
                activity_regularizer=keras.regularizers.l2(1e-9),
            )

        return keras.Sequential(
            [
                layer_conv2d(),
                keras.layers.AveragePooling2D((2, 2), padding="same"),
                layer_conv2d(),
                layer_conv2d(),
                keras.layers.UpSampling2D((2, 2)),
                layer_conv2d(),
                layer_conv2d(),
            ]
        )

    def pre_train(self):
        with self.tensorboard_file_writer().as_default():
            x = np.concatenate([x for x, y in self.data_test.take(1)], axis=0)
            tf.summary.image(f"{self.name()} test input", x, step=0)

    def post_train(self):
        pass

    def custom_callbacks(self) -> List[keras.callbacks.Callback]:
        def predict(epoch, logs):
            with self.tensorboard_file_writer().as_default():
                tf.summary.image(
                    f"{self.name()} (epoch: {epoch}) test prediction",
                    self.model().predict(self.data_test.take(1)),
                    step=epoch,
                )

        return [keras.callbacks.LambdaCallback(on_epoch_end=predict)]


if __name__ == "__main__":
    train_set, test_set = NoisyMnist()
    reformer = Reformer(train_set, test_set, (28, 28, 1))
    reformer.train()
