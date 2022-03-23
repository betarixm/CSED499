from abc import ABC, abstractmethod

import tensorflow as tf

keras = tf.keras


class Model(ABC):
    def __init__(
        self,
        data_train: tf.data.Dataset,
        data_test: tf.data.Dataset,
        input_shape: tuple,
        optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(),
        loss: keras.losses.Loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
        accuracy: keras.metrics.Accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        ),
        checkpoint_filepath: str = None,
    ):
        self._name: str = self.name()

        self.data_train: tf.data.Dataset = data_train
        self.data_test: tf.data.Dataset = data_test

        self.input_shape = input_shape

        self._model: keras.Model = keras.Sequential(
            [keras.Input(self.input_shape), *self.model().layers], name=self._name
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

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def model(self) -> keras.Model:
        pass

    def train(self, epochs: int = 100):
        self._model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=[self.accuracy],
        )

        try:
            self._model.load_weights(self.checkpoint_filepath)
        except tf.errors.NotFoundError:
            pass

        self._model.summary()
        self._model.fit(
            self.data_train,
            epochs=epochs,
            validation_data=self.data_test,
            callbacks=[self.checkpoint_callback],
        )

        self._model.evaluate(self.data_test)
