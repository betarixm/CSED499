from utils.dataset import NoisyMnist, NoisyCifar10
from models import Reformer

import tensorflow as tf

keras = tf.keras


def train_mnist_reformer():
    train_set, test_set = NoisyMnist().dataset()
    reformer = Reformer(train_set, test_set, (28, 28, 1), name="defense_reformer_mnist")
    reformer.train()


def train_cifar10_reformer():
    train_set, test_set = NoisyCifar10().dataset()
    reformer = Reformer(
        train_set,
        test_set,
        (32, 32, 3),
        name="defense_reformer_cifar10",
        accuracy=keras.metrics.CategoricalAccuracy(name="accuracy"),
    )
    reformer.train()


if __name__ == "__main__":
    train_mnist_reformer()
