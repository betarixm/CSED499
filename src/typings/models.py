from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from typings.dataset import Dataset

import datetime

import tensorflow as tf
import numpy as np

keras = tf.keras


class Model(ABC):
    def __init__(
        self,
        name: str,
        input_shape: tuple,
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


class Attack(ABC):
    def __init__(
        self,
        victim_model: Model,
        dataset: Dataset,
        defense_model: Model = None,
        accuracy_normal: keras.metrics.Accuracy = tf.metrics.SparseCategoricalAccuracy(),
        accuracy_under_attack: keras.metrics.Accuracy = tf.metrics.SparseCategoricalAccuracy(),
        accuracy_with_defense: keras.metrics.Accuracy = tf.metrics.SparseCategoricalAccuracy(),
    ):
        self.victim_model: Model = victim_model
        self.defense_model: Model = defense_model
        self.dataset: Dataset = dataset
        self.accuracy_normal: tf.metrics.Accuracy = accuracy_normal
        self.accuracy_under_attack: tf.metrics.Accuracy = accuracy_under_attack
        self.accuracy_with_defense: tf.metrics.Accuracy = accuracy_with_defense

        self.victim_model.compile()
        self.victim_model.load()

        if defense_model is not None:
            self.defense_model.compile()
            self.defense_model.load()

    @abstractmethod
    def add_perturbation(self, x: np.array) -> np.array:
        pass

    def attack(
        self,
    ) -> Tuple[tf.metrics.Accuracy, tf.metrics.Accuracy, Optional[tf.metrics.Accuracy]]:
        _, test = self.dataset.dataset()

        progress = keras.utils.Progbar(test.cardinality().numpy())

        for x, y in test:
            x_attack = self.add_perturbation(x)

            y_attack = self.victim_model.predict(x_attack)
            y_normal = self.victim_model.predict(x)

            self.accuracy_under_attack(y, y_attack)
            self.accuracy_normal(y, y_normal)

            if self.defense_model is not None:
                self.accuracy_with_defense(
                    y, self.victim_model.predict(self.defense_model.predict(x_attack))
                )
            progress.add(1)

        return (
            self.accuracy_normal,
            self.accuracy_under_attack,
            self.accuracy_with_defense,
        )
