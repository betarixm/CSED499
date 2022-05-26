from utils.dataset import NoisyMnist, NoisyCifar10, ExCifar10, TgCifar10
from models import Reformer, Exformer

import argparse
import tensorflow as tf

keras = tf.keras


def train_mnist_reformer(epochs: int = 100):
    train_set, test_set = NoisyMnist().dataset()
    reformer = Reformer(
        "defense_reformer_mnist",
        (28, 28, 1),
        1.0,
        train_set,
        test_set,
    )
    reformer.train(epochs)


def train_cifar10_reformer(epochs: int = 100):
    train_set, test_set = NoisyCifar10().dataset()
    reformer = Reformer(
        "defense_reformer_cifar10",
        (32, 32, 3),
        1.0,
        train_set,
        test_set,
    )
    reformer.train(epochs)


def train_excifar10_exformer(epochs: int = 100):
    train_set, test_set = ExCifar10().dataset()
    exformer = Exformer(
        "defense_exformer_cifar10",
        (32, 32, 3),
        1.0,
        train_set,
        test_set,
    )
    exformer.train(epochs)


def train_tgcifar10_tgformer(epochs: int = 100):
    train_set, test_set = TgCifar10().dataset()
    tgformer = Exformer(
        "defense_tgformer_cifar10",
        (32, 32, 3),
        1.0,
        train_set,
        test_set,
    )
    tgformer.train(epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training defensive reformer models.")
    parser.add_argument(
        "--dataset",
        "-d",
        metavar="DATASET",
        type=str,
        help="Dataset for training",
        required=True,
        choices=["mnist", "cifar10", "excifar10", "tgcifar10"],
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
        train_mnist_reformer(e)
    elif args.dataset == "cifar10":
        train_cifar10_reformer(e)
    elif args.dataset == "excifar10":
        train_excifar10_exformer(e)
    elif args.dataset == "tgcifar10":
        train_tgcifar10_tgformer(e)
