from utils.dataset import NoisyMnist, NoisyCifar10
from models import Reformer

import argparse
import tensorflow as tf

keras = tf.keras


def train_mnist_reformer(epochs: int = 100):
    train_set, test_set = NoisyMnist().dataset()
    reformer = Reformer(train_set, test_set, (28, 28, 1), name="defense_reformer_mnist")
    reformer.train(epochs)


def train_cifar10_reformer(epochs: int = 100):
    train_set, test_set = NoisyCifar10().dataset()
    reformer = Reformer(
        train_set,
        test_set,
        (32, 32, 3),
        name="defense_reformer_cifar10",
        accuracy=keras.metrics.CategoricalAccuracy(name="accuracy"),
    )
    reformer.train(epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training defensive reformer models.")
    parser.add_argument(
        "--dataset",
        "-d",
        metavar="DATASET",
        type=str,
        help="Dataset for training",
        required=True,
        choices=["mnist", "cifar10"],
    )

    parser.add_argument(
        "--epochs",
        "-e",
        metavar="EPOCHS",
        type=int,
        help="Target epochs",
        required=False,
    )

    args = parser.parse_args()

    epochs = args.epochs if args.epochs is not None else 500

    if args.dataset == "mnist":
        train_mnist_reformer(epochs)
    elif args.dataset == "cifar10":
        train_cifar10_reformer(epochs)
