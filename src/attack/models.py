from typings.models import Attack

from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent

import numpy as np
import tensorflow as tf

keras = tf.keras


class Fgsm(Attack):
    def add_perturbation(self, x: np.array) -> np.array:
        return fast_gradient_method(self.victim_model.model(), x, 0.3, np.inf)


class Pgd(Attack):
    def add_perturbation(self, x: np.array) -> np.array:
        return projected_gradient_descent(
            self.victim_model.model(), x, 0.05, 0.01, 40, np.inf
        )
