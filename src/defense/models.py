from typing import List, Literal, Tuple
from typings.models import Defense
from utils.layers import SlqLayer

import numpy as np
import tensorflow as tf

keras = tf.keras


class Reformer(Defense):
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
                keras.layers.Conv2D(
                    self.input_shape()[-1],
                    (3, 3),
                    activation="sigmoid",
                    padding="same",
                    activity_regularizer=keras.regularizers.l2(1e-9),
                ),
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


class Denoiser(Defense):
    def _model(self) -> keras.Model:
        return keras.Sequential([SlqLayer()])

    def pre_train(self):
        pass

    def post_train(self):
        pass

    def custom_callbacks(self) -> List[keras.callbacks.Callback]:
        pass


class Motd(Defense):
    def __init__(
        self,
        name: str,
        input_shape: tuple,
        dataset: Literal["mnist", "cifar10"],
        intensities: Tuple[float, float] = (1.0, 1.0),
        *args,
        **kwargs,
    ):
        self.denoiser = Denoiser(
            f"defense_denoiser_{dataset}",
            input_shape=input_shape,
            intensity=intensities[0],
        )
        self.reformer = Reformer(
            f"defense_reformer_{dataset}",
            input_shape=input_shape,
            intensity=intensities[1],
        )

        super().__init__(name, input_shape, *args, **kwargs)

    def _model(self) -> keras.Model:
        self.reformer.compile()
        self.reformer.load()

        return keras.Model(
            self.denoiser.model().inputs,
            self.reformer.model()(self.denoiser.model().outputs),
        )

    def pre_train(self):
        pass

    def post_train(self):
        pass

    def custom_callbacks(self) -> List[keras.callbacks.Callback]:
        pass
