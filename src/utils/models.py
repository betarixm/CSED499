from typings.models import Model
from typing import List
from functools import reduce

import tensorflow as tf

keras = tf.keras


class SequentialInternalModel(keras.Model):
    def __init__(self, models: List[Model], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = models

    def call(self, inputs, training=None, mask=None):
        return reduce(
            lambda x, y: y(x), reversed([m.predict for m in self.models]), inputs
        )

    def get_config(self):
        pass
