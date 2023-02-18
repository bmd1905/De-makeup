import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
from yaml.loader import SafeLoader

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from model.trainer import Trainer
from model.resnet34_unet import ReNet34_UNet
from loader.dataloader import Dataloader
#/Users/bmd1905/MyDocuments/Project/De-makeup/tools/train.py
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--img", help="File path of image.")
    # parser.add_argument("--weights", help="Weights of the model.", default='./pretrained/de_makeup_50_25psnr_epoch.h5')

    # args = parser.parse_args()
    # config = args
    # Load config file
    # import pyyaml module


    # Open the file and load the file
    with open('config.yml') as f:
        config = yaml.load(f, Loader=SafeLoader)


    # For train from scratch
    # ob = ReNet34_UNet((224, 224, 3))
    # model = ob.build_model()
    #model.load_weights('./pretrained/de_makeup_30_epoch.h5')
    
    # For fine-tuning
    # load model
    #model = tf.keras.models.load_model('pretrained/de_makeup_epochs_26_3psnr_epoch.h5')
    

    # Configuration model
    trainer = Trainer(config)

    # Load data
    dataloader = Dataloader(config)
    train_ds, val_ds, test_ds = dataloader.loader()

    # Train
    trainer.train(train_ds, val_ds)

