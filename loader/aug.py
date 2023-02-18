import tensorflow as tf
import imgaug.augmenters as iaa

class ImgAugTransform:
    def __init__(self):
        self.augmenter = iaa.Sequential([
            iaa.Lambda(
                func_images=self.random_flip,
                #arguments={"non_img": None},
            ),
        ])

    @staticmethod
    @tf.function
    def random_flip(makeup_img, non_img):
        if tf.random.uniform(()) > 0.5:
            # Random mirroring
            makeup_img = tf.image.flip_left_right(makeup_img)
            non_img = tf.image.flip_left_right(non_img)

        return makeup_img, non_img

    def __call__(self, makeup_img, non_img):
        images = self.augmenter(images=tf.convert_to_tensor([makeup_img]))[0]
        return images.numpy(), non_img
