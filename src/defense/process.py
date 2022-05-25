from utils.logging import concat_batch_images
from defense.models import Reformer, Denoiser, Motd, ExMotd

import tensorflow as tf

keras = tf.keras

if __name__ == "__main__":
    from utils.dataset import Mnist, Cifar10
    import argparse

    parser = argparse.ArgumentParser(
        description="Logging images generated by defense models"
    )

    parser.add_argument(
        "--dataset",
        "-d",
        metavar="DATASET",
        type=str,
        help="Dataset for testing",
        required=True,
        choices=["mnist", "cifar10"],
    )

    parser.add_argument(
        "--defense",
        "-f",
        metavar="DEFENSE",
        type=str,
        help="Defense method",
        required=True,
        choices=["reformer", "exformer", "denoiser", "motd", "exmotd"],
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

    if args.dataset == "mnist":
        input_shape = (28, 28, 1)
        _, test_set = Mnist().dataset()

    else:
        input_shape = (32, 32, 3)
        _, test_set = Cifar10().dataset()

    if args.defense == "reformer":
        defense_model = Reformer(
            f"defense_reformer_{args.dataset}",
            input_shape=input_shape,
            intensity=args.intensity[0],
        )
    elif args.defense == "exformer":
        defense_model = Reformer(
            "defense_exformer_cifar10",
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
        defense_model = ExMotd(
            f"defense_exmotd_{args.dataset}",
            input_shape=input_shape,
            dataset=args.dataset,
            intensities=args.intensity,
        )

    defense_model.compile()
    defense_model.load()

    progress = keras.utils.Progbar(test_set.cardinality().numpy())

    with tf.summary.create_file_writer(
        f"./logs/defense_processing_result"
    ).as_default():
        for idx, (x, _) in enumerate(test_set):
            y = defense_model.predict(x)

            tf.summary.image(
                f"(Defense)[{args.dataset.upper()}] Original images",
                concat_batch_images(x),
                step=idx,
            )

            tf.summary.image(
                f"(Defense)[{args.dataset.upper()}] {args.defense.upper()} ({', '.join([str(_) for _ in args.intensity])}) processing result",
                concat_batch_images(y),
                step=idx,
            )

            progress.add(1)
