from typings.models import Attack
from models import Fgsm
from defense.models import Reformer
from victim.models import Classifier

from utils.dataset import Mnist, Cifar10

import argparse
import tensorflow as tf


keras = tf.keras

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Attack and defense pretrained models."
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
        "--defense",
        "-f",
        help="Use defense model",
        action="store_true",
    )

    args = parser.parse_args()

    a: Attack

    if args.dataset == "mnist":
        a = Fgsm(
            Classifier(name="victim_classifier_mnist", input_shape=(28, 28, 1)),
            Mnist(),
            defense_model=Reformer("defense_reformer_mnist", input_shape=(28, 28, 1))
            if args.defense
            else None,
        )
    elif args.dataset == "cifar10":
        a = Fgsm(
            Classifier(name="victim_classifier_cifar10", input_shape=(32, 32, 3)),
            Cifar10(),
            defense_model=Reformer("defense_reformer_cifar10", input_shape=(32, 32, 3))
            if args.defense
            else None,
        )

    acc, acc_under_attack, acc_with_defense = a.attack()

    print(f"[*] Attack {args.dataset.upper()} {'with defense' if args.defense else ''}")
    print(f"    - Normal:       {acc.result()}")
    print(f"    - Under Attack: {acc_under_attack.result()}")

    if args.defense:
        print(f"    - With Defense: {acc_with_defense.result()}")
