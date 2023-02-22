import matplotlib.pyplot as plt
import argparse

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import load_img


if __name__ == '__main__':
    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", help="File path of image.")
    parser.add_argument("--weights", help="Weights of the model.", default='pretrained/de_makeup_epochs_26_3psnr_epoch.h5')

    args = parser.parse_args()
    
    # Load model
    model = tf.keras.models.load_model(args.weights)
    
    # Load the image from file
    img_path = args.img
    img_name = img_path.split('/')[-1]
    img = load_img(img_path, target_size=(224, 224, 3))

    # Convert the image to a tensor
    img_tensor = tf.keras.preprocessing.image.img_to_array(img)
    img_tensor = tf.reshape(img_tensor, (1, 224, 224, 3))

    output = model(img_tensor)
    output = tf.reshape(output, (224, 224, 3))
    # Convert the tensor to an image
    output_image = tf.keras.preprocessing.image.array_to_img(output)

    # Save image
    output_image.save(f'./runs/predict/predicted_{img_name}')

    # Display the image using Matplotlib
    # plt.imshow(output_image)
    # plt.show()