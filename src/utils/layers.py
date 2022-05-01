import tensorflow as tf

keras = tf.keras


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
                keras.layers.experimental.preprocessing.Rescaling(255)(image),
                tf.uint8,
            )

            n, m, c = image.shape

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

            return keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(
                tf.gather_nd(x_compressed_stack, indices, name="final_image")
            )

        return tf.map_fn(fn=compress, elems=inputs)
