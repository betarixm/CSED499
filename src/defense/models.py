from typings.models import Model
from utils.dataset import NoisyMnist

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

    def name(self) -> str:
        return "defense_reformer"

    def model(self) -> keras.Model:
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


if __name__ == "__main__":
    train_set, test_set = NoisyMnist()
    reformer = Reformer(train_set, test_set, (28, 28, 1))
    reformer.train()
