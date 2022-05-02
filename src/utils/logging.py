import numpy as np
import tensorflow as tf


def concat_batch_images(batch_images: np.array):
    width = 4
    return tf.expand_dims(
        np.concatenate(
            [
                np.concatenate(i, axis=1)
                for i in batch_images.reshape(-1, width, *batch_images.shape[1:])
            ],
            axis=0,
        ),
        0,
    )
