from typing import List
from typings.models import Model

import numpy as np
import tensorflow as tf

keras = tf.keras


class Classifier(Model):
    def _model(self) -> keras.Model:
        return keras.Sequential(
            [
                keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.2),
                keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.3),
                keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.4),
                keras.layers.Flatten(),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(10, activation="softmax"),
            ],
        )

    def pre_train(self):
        with self.tensorboard_file_writer().as_default():
            x = np.concatenate([x for x, y in self.data_test.take(1)], axis=0)
            tf.summary.image(f"{self.name()} test input", x, max_outputs=25, step=0)

    def post_train(self):
        pass

    def custom_callbacks(self) -> List[keras.callbacks.Callback]:
        return []
