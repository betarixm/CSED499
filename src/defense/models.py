from typing import List, Literal
from typings.models import Model
from utils.layers import SlqLayer
from utils.logging import plot_to_image, batch_image_grid

import numpy as np
import tensorflow as tf

keras = tf.keras


class Reformer(Model):
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


class Denoiser(Model):
    def _model(self) -> keras.Model:
        return keras.Sequential([SlqLayer()])

    def pre_train(self):
        pass

    def post_train(self):
        pass

    def custom_callbacks(self) -> List[keras.callbacks.Callback]:
        pass


class Motd(Model):
    def __init__(
        self,
        name: str,
        input_shape: tuple,
        dataset: Literal["mnist", "cifar10"],
        **kwargs,
    ):
        self.reformer = Reformer(f"defense_reformer_{dataset}", input_shape=input_shape)
        self.denoiser = Denoiser(f"defense_denoiser_{dataset}", input_shape=input_shape)

        super().__init__(name, input_shape, **kwargs)

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


if __name__ == "__main__":
    from utils.dataset import Mnist, Cifar10
    import argparse

    parser = argparse.ArgumentParser(
        description="Logging images generated by defense models"
    )

    parser.add_argument(
        "--dataset",
        "-d",
        metavar="DATASET",
        type=str,
        help="Dataset for testing",
        required=True,
        choices=["mnist", "cifar10"],
    )

    parser.add_argument(
        "--defense",
        "-f",
        metavar="DEFENSE",
        type=str,
        help="Defense method",
        required=True,
        choices=["reformer", "denoiser", "motd"],
    )

    args = parser.parse_args()

    if args.dataset == "mnist":
        input_shape = (28, 28, 1)
        _, test_set = Mnist().dataset()

    else:
        input_shape = (32, 32, 3)
        _, test_set = Cifar10().dataset()

    if args.defense == "reformer":
        defense_model = Reformer(
            f"defense_reformer_{args.dataset}", input_shape=input_shape
        )
    elif args.defense == "denoiser":
        defense_model = Denoiser(
            f"defense_denoiser_{args.dataset}", input_shape=input_shape
        )
    else:
        defense_model = Motd(
            f"defense_motd_{args.dataset}",
            input_shape=input_shape,
            dataset=args.dataset,
        )

    defense_model.compile()
    defense_model.load()

    progress = keras.utils.Progbar(test_set.cardinality().numpy())

    with defense_model.tensorboard_file_writer().as_default():
        for idx, (x, _) in enumerate(test_set):
            y = defense_model.predict(x)

            tf.summary.image(
                f"(Defense) Original images",
                plot_to_image(batch_image_grid(x)),
                step=idx,
            )

            tf.summary.image(
                f"(Defense) {args.defense.upper()} processing result",
                plot_to_image(batch_image_grid(y)),
                step=idx,
            )

            progress.add(1)
