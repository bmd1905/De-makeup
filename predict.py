import numpy as np
import matplotlib.pyplot as plt
import argparse

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from model.resnet34_unet import ReNet34_UNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", help="File path of image.")
    parser.add_argument("--weights", help="Weights of the model.", default='./pretrained/de_makeup_30_epoch.h5')

    args = parser.parse_args()
    # For cmd testing
    # Create model
    ob = ReNet34_UNet((224, 224, 3))
    model = ob.build_model()
    
    #model.load_weights('./pretrained/de_makeup_30_epoch.h5')
    model.load_weights(args.weights)
    
    # load the image from file
    # img = load_img('./sample/makeup1.jpg', target_size=(224, 224, 3))
    img = load_img(args.img, target_size=(224, 224, 3))

    # convert the image to a tensor
    img_tensor = tf.keras.preprocessing.image.img_to_array(img)
    img_tensor = tf.reshape(img_tensor, (1, 224, 224, 3))

    output = model(img_tensor)
    output = tf.reshape(output, (224, 224, 3))
    # convert the tensor to an image
    output_image = tf.keras.preprocessing.image.array_to_img(output)

    # display the image using Matplotlib
    plt.imshow(output_image)
    plt.show()