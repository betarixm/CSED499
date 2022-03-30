from typings.models import Attack, Model
from typings.dataset import Dataset

from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

from victim.models import Classifier
from utils.dataset import Cifar10

import numpy as np
import tensorflow as tf


keras = tf.keras


class Fgsm(Attack):
    def __init__(
        self,
        model: Model,
        dataset: Dataset,
        accuracy_normal: keras.metrics.Accuracy = tf.metrics.SparseCategoricalAccuracy(),
        accuracy_under_attack: keras.metrics.Accuracy = tf.metrics.SparseCategoricalAccuracy(),
    ):
        super(Fgsm, self).__init__(
            model, dataset, accuracy_normal, accuracy_under_attack
        )

    def add_perturbation(self, x: np.array) -> np.array:
        return fast_gradient_method(self.model.model(), x, 0.05, np.inf)


if __name__ == "__main__":
    f = Fgsm(
        Classifier(name="victim_classifier_cifar10", input_shape=(32, 32, 3)), Cifar10()
    )
    acc_with_attack, acc = f.attack()
    print(acc_with_attack.result(), acc.result())
