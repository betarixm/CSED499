from tensorflow.python.keras import Sequential
from tensorflow.python.keras import layers

import tensorflow as tf


class Mnist:
    model = Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ],
        name="victim_mnist",
    )

    model.summary()

    def __init__(
        self,
        optimizer=tf.keras.optimizers.Adam(),
        train_loss=tf.keras.metrics.Mean(name="train_loss"),
        train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy"
        ),
        test_loss=tf.keras.metrics.Mean(name="test_loss"),
        test_accuracy=tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy"),
        loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    ):
        self.optimizer = optimizer
        self.train_loss = train_loss
        self.train_accuracy = train_accuracy

        self.test_loss = test_loss
        self.test_accuracy = test_accuracy

        self.loss_object = loss_object

    def dataset(self):
        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        x_train = x_train[..., tf.newaxis].astype("float32")
        x_test = x_test[..., tf.newaxis].astype("float32")

        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(10000)
            .batch(32)
        )

        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

        return train_ds, test_ds

    def train(self, epochs: int = 100):
        train_ds, test_ds = self.dataset()
        self.model.fit(train_ds, epochs=epochs)
        self.model.evaluate(test_ds)


if __name__ == "__main__":
    mnist = Mnist()
    mnist.train()
