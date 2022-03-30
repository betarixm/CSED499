from utils.dataset import Mnist, Cifar10
from models import Classifier

import argparse
import tensorflow as tf

keras = tf.keras


def train_mnist_classifier(epochs: int = 100):
    train_set, test_set = Mnist().dataset()
    classifier = Classifier(
        "victim_classifier_mnist",
        (28, 28, 1),
        train_set,
        test_set,
    )
    classifier.train(epochs)


def train_cifar10_classifier(epochs: int = 100):
    train_set, test_set = Cifar10().dataset()
    reformer = Classifier(
        "victim_classifier_cifar10",
        (32, 32, 3),
        train_set,
        test_set,
    )
    reformer.train(epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training victim classifier models.")
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

    e = args.epochs if args.epochs is not None else 500

    if args.dataset == "mnist":
        train_mnist_classifier(e)
    elif args.dataset == "cifar10":
        train_cifar10_classifier(e)
