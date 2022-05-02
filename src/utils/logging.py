import io
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def batch_image_grid(batch_images: np.array):
    width = 8
    size = batch_images.shape[0]
    figure = plt.figure(figsize=(10, 10))

    for idx, image in enumerate(batch_images):
        plt.subplot(width, math.ceil(size / width), idx + 1)
        plt.xticks([]), plt.yticks([])
        plt.grid(False)
        plt.imshow(image)

    return figure


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image
