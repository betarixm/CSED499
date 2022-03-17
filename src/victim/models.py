from tensorflow.python.keras import Model
from tensorflow.python.keras import layers

import tensorflow as tf


class Mnist:
    class Network(Model):
        def __init__(self):
            super(self.__class__, self).__init__()
            self.conv1 = layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(28, 28, 1)
            )
            self.conv2 = layers.Conv2D(64, (3, 3), activation="relu")
            self.max_pooling = layers.MaxPooling2D((2, 2))

        def call(self, x):
            x = self.conv1(x)
            x = self.max_pooling(x)
            x = self.conv2(x)
            x = self.max_pooling(x)
            x = self.conv2(x)
            x = layers.Flatten()(x)
            x = layers.Dense(64, activation="relu")(x)
            x = layers.Dense(10, activation="softmax")(x)
            return x

    model = Network()

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
        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                predictions = self.model(images, training=True)
                loss = self.loss_object(labels, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )

            self.train_loss(loss)
            self.train_accuracy(labels, predictions)

        @tf.function
        def test_step(images, labels):
            predictions = self.model(images, training=False)
            t_loss = self.loss_object(labels, predictions)

            self.test_loss(t_loss)
            self.test_accuracy(labels, predictions)

        train_ds, test_ds = self.dataset()

        for epoch in range(epochs):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states(), self.train_accuracy.reset_states()
            self.test_loss.reset_states(), self.test_accuracy.reset_states()

            for images, labels in train_ds:
                train_step(images, labels)

            for test_images, test_labels in test_ds:
                test_step(test_images, test_labels)

            print(
                f"Epoch {epoch + 1}, "
                f"Loss: {self.train_loss.result()}, "
                f"Accuracy: {self.train_accuracy.result() * 100}, "
                f"Test Loss: {self.test_loss.result()}, "
                f"Test Accuracy: {self.test_accuracy.result() * 100}"
            )


if __name__ == "__main__":
    mnist = Mnist()
    mnist.train()
