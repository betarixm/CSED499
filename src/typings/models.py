from abc import ABC, abstractmethod
from typing import List

import datetime

import tensorflow as tf

keras = tf.keras


class Model(ABC):
    def __init__(
        self,
        name: str,
        input_shape: tuple,
        intensity: float = 1.0,
        data_train: tf.data.Dataset = None,
        data_test: tf.data.Dataset = None,
        optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(),
        loss: keras.losses.Loss = keras.losses.SparseCategoricalCrossentropy(),
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

        self.__model: keras.Model = (
            keras.Sequential(
                [keras.Input(self.__input_shape), *self._model().layers],
                name=self._name,
            )
            if len(self._model().layers) != 0
            else self._model()
        )

        self.intensity: float = intensity

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

    def pre_train(self):
        pass

    def post_train(self):
        pass

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
        try:
            self.__model.load_weights(self.checkpoint_filepath)
        except tf.errors.NotFoundError:
            pass

    def compile(self):
        self.__model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=[self.accuracy],
        )

    def predict(self, inputs):
        outs = self.__model(inputs)
        return inputs + (outs - inputs) * self.intensity if self.intensity < 1 else outs

    def train(self, epochs: int = 100):
        self.pre_train()

        self.compile()

        self.load()

        self.__model.summary()
        self.__model.fit(
            self.data_train,
            epochs=epochs,
            validation_data=self.data_test,
            callbacks=[
                self.checkpoint_callback,
                self.tensorboard_callback,
                keras.callbacks.EarlyStopping(monitor="loss", patience=5),
                *self.custom_callbacks(),
            ],
        )

        self.post_train()

    def evaluate(self):
        self.compile()
        self.load()
        result = self.__model.evaluate(self.data_test)
        return dict(zip(self.__model.metrics_names, result))


class Defense(Model, ABC):
    pass


class Attack(Model, ABC):
    def __init__(
        self,
        name: str,
        input_shape: tuple,
        victim_model: Model,
        *args,
        **kwargs,
    ):
        self.victim_model: Model = victim_model
        super().__init__(name, input_shape, *args, **kwargs)
