import numpy as np
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2, CarliniWagnerL2

import tensorflow as tf

keras = tf.keras


class FgsmLayer(keras.layers.Layer):
    def __init__(self, victim_model: keras.Model, eps: float, norm: float, **kwargs):
        super().__init__(**kwargs)
        self.victim_model = victim_model
        self.eps = eps
        self.norm = norm

    def call(self, inputs, *args, **kwargs):
        return fast_gradient_method(self.victim_model, inputs, self.eps, self.norm)


class PgdLayer(keras.layers.Layer):
    def __init__(
        self,
        victim_model: keras.Model,
        eps: float,
        step: float,
        batch_size: int,
        norm: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.victim_model: keras.Model = victim_model
        self.eps: float = eps
        self.step: float = step
        self.batch_size: int = batch_size
        self.norm: float = norm

    def call(self, inputs, *args, **kwargs):
        return projected_gradient_descent(
            self.victim_model,
            tf.cast(inputs, tf.float32),
            self.eps,
            self.step,
            self.batch_size,
            self.norm,
        )


class CwLayer(keras.layers.Layer):
    def __init__(
        self,
        victim_model: keras.Model,
        batch_size: int,
        clip_min: float,
        clip_max: float,
        binary_search_steps: int,
        max_iterations: int,
        initial_const: int,
        learning_rate: float,
        **kwargs
    ):

        super().__init__(**kwargs)
        self.carlini_wagner = CarliniWagnerL2(
            victim_model,
            batch_size=batch_size,
            clip_min=clip_min,
            clip_max=clip_max,
            binary_search_steps=binary_search_steps,
            max_iterations=max_iterations,
            initial_const=initial_const,
            learning_rate=learning_rate,
        )

    def call(self, inputs, *args, **kwargs):
        return tf.numpy_function(
            self.carlini_wagner.attack, [tf.cast(inputs, tf.float32)], tf.float32
        )


class NormalNoiseLayer(keras.layers.Layer):
    def __init__(self, intensity: float, **kwargs):
        super().__init__(**kwargs)
        self.intensity: float = intensity

    def call(self, inputs, *args, **kwargs):
        def add_noise(x, intensity):
            return x + intensity * np.random.normal(size=np.shape(x))

        return tf.clip_by_value(
            tf.numpy_function(
                add_noise, [tf.cast(inputs, tf.float32), self.intensity], tf.float32
            ),
            0.0,
            1.0,
        )


class SlqLayer(keras.layers.Layer):
    """
    Das, Nilaksh, et al.
    "Shield: Fast, practical defense and vaccination for deep learning using jpeg compression."
    Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018.
    """

    def __init__(self, qualities=(20, 40, 60, 80), patch_size=8, **kwargs):
        super().__init__(**kwargs)
        self.qualities = qualities
        self.patch_size = patch_size

    @tf.function
    def call(self, inputs, *args, **kwargs):
        def compress(image):
            one = tf.constant(1, name="one")
            zero = tf.constant(0, name="zero")

            image = tf.cast(
                keras.layers.experimental.preprocessing.Rescaling(255)(
                    tf.clip_by_value(image, 0.0, 1.0)
                ),
                tf.uint8,
            )

            n, m, c = image.shape

            if c == 1:
                image = tf.image.grayscale_to_rgb(image)

            patch_n = tf.cast(n / self.patch_size, dtype=tf.int32) + tf.cond(
                tf.constant(n % self.patch_size > 0, tf.bool),
                lambda: one,
                lambda: zero,
            )

            patch_m = tf.cast(m / self.patch_size, dtype=tf.int32) + tf.cond(
                tf.constant(m % self.patch_size > 0, tf.bool),
                lambda: one,
                lambda: zero,
            )

            R = tf.tile(tf.reshape(tf.range(n), (n, 1)), [1, m])
            C = tf.reshape(tf.tile(tf.range(m), [n]), (n, m))
            Z = tf.compat.v1.image.resize_nearest_neighbor(
                [
                    tf.compat.v1.random_uniform(
                        (patch_n, patch_m, 3),
                        0,
                        len(self.qualities),
                        dtype=tf.int32,
                    )
                ],
                (patch_n * self.patch_size, patch_m * self.patch_size),
                name="random_layer_indices",
            )[0, :, :, 0][:n, :m]

            indices = tf.transpose(
                tf.stack([Z, R, C]), perm=[1, 2, 0], name="random_layer_indices"
            )

            x_compressed_stack = tf.stack(
                list(
                    map(
                        lambda q: tf.image.decode_jpeg(
                            tf.image.encode_jpeg(image, format="rgb", quality=q),
                            channels=3,
                        ),
                        self.qualities,
                    )
                ),
                name="compressed_images",
            )

            result = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(
                tf.gather_nd(x_compressed_stack, indices, name="final_image")
            )

            if c == 1:
                result = tf.image.rgb_to_grayscale(result)

            return result

        return tf.clip_by_value(tf.map_fn(fn=compress, elems=inputs), 0.0, 1.0)
