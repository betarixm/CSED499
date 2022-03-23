import tensorflow as tf
import numpy as np

keras = tf.keras


class Reformer:
    layer_conv2d = keras.layers.Conv2D(
        3,
        (3, 3),
        activation="sigmoid",
        padding="same",
        activity_regularizer=keras.regularizers.l2(1e-9),
    )

    model = keras.Sequential(
        [
            layer_conv2d,
            keras.layers.AveragePooling2D((2, 2), padding="same"),
            layer_conv2d,
            layer_conv2d,
            keras.layers.UpSampling2D((2, 2)),
            layer_conv2d,
            keras.layers.Conv2D(
                1,
                (3, 3),
                activation="sigmoid",
                padding="same",
                activity_regularizer=keras.regularizers.l2(1e-9),
            ),
        ]
    )

    checkpoint_filepath = "./checkpoint/reformer"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    )

    def __init__(
        self,
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.MeanSquaredError(),
        accuracy=keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    ):
        self.optimizer = optimizer
        self.loss = loss
        self.accuracy = accuracy

    def dataset(self):
        def noisy(ds):
            noise = 0.1 * np.random.normal(size=np.shape(ds))
            return np.clip(ds + noise, 0.0, 1.0)

        mnist = keras.datasets.mnist

        (x_train, _), (x_test, _) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        x_train = x_train[..., tf.newaxis].astype("float32")
        x_test = x_test[..., tf.newaxis].astype("float32")

        x_train_noisy, x_test_noisy = noisy(x_train), noisy(x_test)

        print(x_train_noisy.shape, x_train.shape)
        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train_noisy, x_train))
            .shuffle(10000)
            .batch(32)
        )

        test_ds = tf.data.Dataset.from_tensor_slices((x_test_noisy, x_test)).batch(32)

        return train_ds, test_ds

    def train(self, epochs: int = 100):
        train_ds, test_ds = self.dataset()

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=[self.accuracy],
        )

        try:
            self.model.load_weights(self.checkpoint_filepath)
        except tf.errors.NotFoundError:
            pass

        self.model.fit(
            train_ds,
            epochs=epochs,
            validation_data=test_ds,
            callbacks=[self.checkpoint_callback],
        )
        self.model.evaluate(test_ds)


if __name__ == "__main__":
    reformer = Reformer()
    reformer.train()
