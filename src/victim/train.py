from utils.dataset import Mnist, Cifar10
from models import Classifier

import argparse
import tensorflow as tf

keras = tf.keras

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training or evaluating victim classifier models."
    )

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

    parser.add_argument(
        "--evaluate",
        "-v",
        help="Evaluate trained model.",
        action="store_true",
    )

    args = parser.parse_args()

    e = args.epochs if args.epochs is not None else 500

    if args.dataset == "mnist":
        train_set, test_set = Mnist().dataset()
        classifier = Classifier(
            "victim_classifier_mnist",
            (28, 28, 1),
            train_set,
            test_set,
        )

    elif args.dataset == "cifar10":
        train_set, test_set = Cifar10().dataset()
        classifier = Classifier(
            "victim_classifier_cifar10",
            (32, 32, 3),
            train_set,
            test_set,
        )

    if args.evaluate:
        result = classifier.evaluate()
        print(f"[*] Evaluation of {args.dataset.upper()} Classifier")
        for key in result:
            print(f"    - {key}: {result[key]}")

    else:
        classifier.train(e)
