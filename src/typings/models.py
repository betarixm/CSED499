from abc import ABC, abstractmethod
from typing import List

import datetime

import tensorflow as tf

keras = tf.keras


class Model(ABC):
    def __init__(
        self,
        name: str,
        data_train: tf.data.Dataset = None,
        data_test: tf.data.Dataset = None,
        input_shape: tuple = None,
        optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(),
        loss: keras.losses.Loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
        accuracy: keras.metrics.Accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        ),
        checkpoint_filepath: str = None,
        tensorboard_log_path: str = None,
    ):
        self._name: str = name

        self.data_train: tf.data.Dataset = data_train
        self.data_test: tf.data.Dataset = data_test

        self.__input_shape = input_shape

        self.__model: keras.Model = keras.Sequential(
            [keras.Input(self.__input_shape), *self._model().layers], name=self._name
        )

        self.optimizer: keras.optimizers.Optimizer = optimizer
        self.loss: keras.losses.Loss = loss
        self.accuracy: keras.metrics.Accuracy = accuracy

        self.checkpoint_filepath: str = (
            checkpoint_filepath
            if checkpoint_filepath is not None
            else f"./checkpoint/{self._name}"
        )
        self.checkpoint_callback: keras.callbacks.Callback = (
            keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint_filepath,
                save_weights_only=True,
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
            )
        )

        self.tensorboard_log_path = (
            (
                tensorboard_log_path
                if tensorboard_log_path is not None
                else f"./logs/{self._name}"
            )
            + "/"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        self.tensorboard_callback: keras.callbacks.Callback = (
            keras.callbacks.TensorBoard(
                log_dir=f"{self.tensorboard_log_path}",
                histogram_freq=1,
            )
        )

        self._file_writer: tf.summary.SummaryWriter = tf.summary.create_file_writer(
            self.tensorboard_log_path
        )

    @abstractmethod
    def _model(self) -> keras.Model:
        pass

    @abstractmethod
    def pre_train(self):
        pass

    @abstractmethod
    def post_train(self):
        pass

    @abstractmethod
    def custom_callbacks(self) -> List[keras.callbacks.Callback]:
        pass

    def name(self) -> str:
        return self._name

    def input_shape(self) -> tuple:
        return self.__input_shape

    def model(self) -> keras.Model:
        return self.__model

    def tensorboard_file_writer(self) -> tf.summary.SummaryWriter:
        return self._file_writer

    def load(self):
        self.__model.load_weights(self.checkpoint_filepath)

    def compile(self):
        self.__model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=[self.accuracy],
        )

    def predict(self, inputs):
        return self.__model.predict(inputs)

    def train(self, epochs: int = 100):
        self.pre_train()

        self.compile()

        try:
            self.__model.load_weights(self.checkpoint_filepath)
        except tf.errors.NotFoundError:
            pass

        self.__model.summary()
        self.__model.fit(
            self.data_train,
            epochs=epochs,
            validation_data=self.data_test,
            callbacks=[
                self.checkpoint_callback,
                self.tensorboard_callback,
                *self.custom_callbacks(),
            ],
        )

        self.__model.evaluate(self.data_test)

        self.post_train()
