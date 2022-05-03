from typing import Tuple, Union
from typings.models import Attack, Model
from models import Fgsm, Pgd, Cw
from defense.models import Reformer, Denoiser, Motd
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
        "--method",
        "-m",
        metavar="METHOD",
        type=str,
        help="Attack method",
        required=True,
        choices=["fgsm", "pgd", "cw"],
    )

    parser.add_argument(
        "--defense",
        "-f",
        metavar="DEFENSE",
        type=str,
        help="Defense method",
        required=True,
        choices=["reformer", "denoiser", "motd", "none"],
    )

    parser.add_argument(
        "--intensity",
        "-i",
        metavar="INTENSITY",
        type=float,
        help="Intensity of processing",
        required=True,
        nargs="+",
    )

    args = parser.parse_args()

    attacker: Attack
    input_shape: Tuple[int, int, int]
    attack_cls: Attack.__class__
    defense_model: Union[Model, None]

    if args.dataset == "mnist":
        input_shape = (28, 28, 1)
        classifier = Classifier(name="victim_classifier_mnist", input_shape=input_shape)

        dataset = Mnist()

    else:
        input_shape = (32, 32, 3)
        classifier = Classifier(
            name="victim_classifier_cifar10", input_shape=input_shape
        )

        dataset = Cifar10()

    if args.method == "fgsm":
        attack_cls = Fgsm
    elif args.method == "pgd":
        attack_cls = Pgd
    else:
        attack_cls = Cw

    if args.defense == "reformer":
        defense_model = Reformer(
            f"defense_reformer_{args.dataset}",
            input_shape=input_shape,
            intensity=args.intensity[0],
        )
    elif args.defense == "denoiser":
        defense_model = Denoiser(
            f"defense_denoiser_{args.dataset}",
            input_shape=input_shape,
            intensity=args.intensity[0],
        )
    elif args.defense == "motd":
        defense_model = Motd(
            f"defense_motd_{args.dataset}",
            input_shape=input_shape,
            dataset=args.dataset,
            intensities=args.intensity,
        )
    else:
        defense_model = None

    attacker = attack_cls(classifier, dataset, defense_model)

    acc, acc_under_attack, acc_with_defense = attacker.attack()

    print(
        f"[*] Attack {args.dataset.upper()} by {args.method.upper()} {'with defense' if args.defense else ''}"
    )
    print(f"    - Normal:       {acc.result()}")
    print(f"    - Under Attack: {acc_under_attack.result()}")

    if defense_model is not None:
        print(
            f"    - With {args.defense.upper()} ({', '.join([str(_) for _ in args.intensity])}): {acc_with_defense.result()}"
        )
