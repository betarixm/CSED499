from typings.models import Attack

from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2

import numpy as np
import tensorflow as tf

keras = tf.keras


class Fgsm(Attack):
    def add_perturbation(self, x: np.array) -> np.array:
        return fast_gradient_method(self.victim_model.model(), x, 0.3, np.inf)


class Pgd(Attack):
    def add_perturbation(self, x: np.array) -> np.array:
        return projected_gradient_descent(
            self.victim_model.model(),
            tf.cast(x, tf.float32),
            0.3,
            0.01,
            40,
            np.inf,
        )


class Cw(Attack):
    def add_perturbation(self, x: np.array) -> np.array:
        return carlini_wagner_l2(
            self.victim_model.model(),
            tf.cast(x, tf.float32),
            batch_size=32,
            clip_min=0.0,
            clip_max=1.0,
            binary_search_steps=16,
            max_iterations=1000,
            initial_const=8,
            learning_rate=0.05,
        )
