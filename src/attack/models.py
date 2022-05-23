from typings.models import Attack
from utils.layers import FgsmLayer, PgdLayer, CwLayer, NormalNoiseLayer


import numpy as np
import tensorflow as tf

keras = tf.keras


class FgsmMnist(Attack):
    def _model(self) -> keras.Model:
        return keras.Sequential([FgsmLayer(self.victim_model.model(), 0.3, np.inf)])


class FgsmCifar(Attack):
    def _model(self) -> keras.Model:
        return keras.Sequential(
            [FgsmLayer(self.victim_model.model(), 16 / 255, np.inf)]
        )


class PgdMnist(Attack):
    def _model(self) -> keras.Model:
        return keras.Sequential(
            [
                PgdLayer(
                    self.victim_model.model(),
                    0.3,
                    0.01,
                    40,
                    np.inf,
                )
            ]
        )


class PgdCifar(Attack):
    def _model(self) -> keras.Model:
        return keras.Sequential(
            [
                PgdLayer(
                    self.victim_model.model(),
                    16 / 255,
                    1 / 255,
                    40,
                    np.inf,
                )
            ]
        )


class NormalNoise(Attack):
    def _model(self) -> keras.Model:
        return keras.Sequential([NormalNoiseLayer(1.0)])


class Cw(Attack):
    def _model(self) -> keras.Model:
        return keras.Sequential(
            [
                CwLayer(
                    self.victim_model.model(),
                    batch_size=32,
                    clip_min=0.0,
                    clip_max=1.0,
                    binary_search_steps=16,
                    max_iterations=1000,
                    initial_const=8,
                    learning_rate=0.05,
                )
            ]
        )
