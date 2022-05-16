from typing import Tuple, Optional
from typings.models import Attack, Defense
from utils.logging import concat_batch_images
from models import FgsmMnist, FgsmCifar, PgdMnist, PgdCifar, Cw
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
    defense_model: Optional[Defense]

    accuracy_normal: tf.metrics.Accuracy = tf.metrics.SparseCategoricalAccuracy()
    accuracy_under_attack: tf.metrics.Accuracy = tf.metrics.SparseCategoricalAccuracy()
    accuracy_with_defense: tf.metrics.Accuracy = tf.metrics.SparseCategoricalAccuracy()

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
        attack_cls = FgsmMnist if args.dataset == "mnist" else FgsmCifar
    elif args.method == "pgd":
        attack_cls = PgdMnist if args.dataset == "mnist" else PgdCifar
    else:
        attack_cls = Cw

    if args.defense == "reformer":
        defense_model = Reformer(
            f"defense_reformer_{args.dataset}",
            input_shape=input_shape,
            intensity=args.intensity[0],
        )

        defense_model.compile()
        defense_model.load()
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

    classifier.compile()
    classifier.load()

    attacker = attack_cls(
        f"attack_{args.method}_{args.dataset}",
        victim_model=classifier,
        input_shape=input_shape,
    )

    _, test = dataset.dataset()

    metrics = ["acc_normal", "acc_under_attack", "acc_with_defense"]
    progress = keras.utils.Progbar(test.cardinality().numpy(), stateful_metrics=metrics)

    with tf.summary.create_file_writer(f"./logs/attack_processing_result").as_default():
        for idx, (x, y) in enumerate(test):
            x_attack = attacker.predict(x)

            x_defense = (
                defense_model.predict(x_attack) if defense_model is not None else None
            )

            y_attack = classifier.predict(x_attack)
            y_normal = classifier.predict(x)

            accuracy_under_attack(y, y_attack)
            accuracy_normal(y, y_normal)

            tf.summary.image(
                f"[{args.dataset.upper()}] Original Images",
                concat_batch_images(x),
                step=idx,
            )

            tf.summary.image(
                f"[{args.dataset.upper()}] {args.method.upper()} Attack Images",
                concat_batch_images(x_attack),
                step=idx,
            )

            if defense_model is not None:
                accuracy_with_defense(y, classifier.predict(x_defense))

                tf.summary.image(
                    f"[{args.dataset.upper()}] {args.method.upper()} Defense Images with {args.defense.upper()} ({', '.join([str(_) for _ in args.intensity])})",
                    concat_batch_images(x_defense),
                    step=idx,
                )

            progress.add(
                1,
                values=zip(
                    metrics,
                    [
                        accuracy_normal.result(),
                        accuracy_under_attack.result(),
                        accuracy_with_defense.result(),
                    ],
                ),
            )

    print(
        f"[*] Attack {args.dataset.upper()} by {args.method.upper()} {'with defense' if args.defense else ''}"
    )
    print(f"    - Normal:       {accuracy_normal.result()}")
    print(f"    - Under Attack: {accuracy_under_attack.result()}")

    if defense_model is not None:
        print(
            f"    - With {args.defense.upper()} ({', '.join([str(_) for _ in args.intensity])}): {accuracy_with_defense.result()}"
        )
