from typings.models import Attack

from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

from victim.models import Classifier
from defense.models import Reformer

from utils.dataset import Cifar10

import numpy as np
import tensorflow as tf

keras = tf.keras


class Fgsm(Attack):
    def add_perturbation(self, x: np.array) -> np.array:
        return fast_gradient_method(self.victim_model.model(), x, 0.05, np.inf)


if __name__ == "__main__":
    f = Fgsm(
        Classifier(name="victim_classifier_cifar10", input_shape=(32, 32, 3)),
        Cifar10(),
        defense_model=Reformer("defense_reformer_cifar10", input_shape=(32, 32, 3)),
    )
    acc, acc_under_attack, acc_with_defense = f.attack()
    print(acc.result(), acc_under_attack.result(), acc_with_defense.result())
