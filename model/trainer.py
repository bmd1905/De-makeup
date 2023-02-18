
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
tf.config.run_functions_eagerly(True)
import sys

from loader.aug import ImgAugTransform

from model.resnet34_unet import ReNet34_UNet
#from official.nlp import optimization  # to create AdamW optimizer
#from tensorflow_addons.optimizers import AdamW
#from tensorflow.keras.optimizers.experimental import AdamW

class Trainer():
    def __init__(self, config, pretrained=True, augmentor=ImgAugTransform()):

        self.config = config
        self.model = ReNet34_UNet(input_shape=(224, 224, 3))
        

        self.epochs =  config['training']['epochs']
        self.init_lr = config['training']['init_lr']
        self.batch_size = config['training']['batch_size']

        # if pretrained:
        #     weight_file = download_weights(config['pretrain'], quiet=config['quiet'])
        #     self.load_weights(weight_file)


        # Optimizer
        # steps_per_epoch = tf.data.experimental.cardinality(self.train_dataset).numpy()
        # num_train_steps = steps_per_epoch * self.epochs
        # num_warmup_steps = int(0.1*num_train_steps)

        # self.optimizer = optimization.create_optimizer(init_lr=self.init_lr,
        #                                         num_train_steps=num_train_steps,
        #                                         num_warmup_steps=num_warmup_steps,
        #                                         optimizer_type='adamw')

        # step = tf.Variable(0, trainable=False)
        # schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
        #     [10000, 15000], [1e-0, 1e-1, 1e-2])
        # # lr and wd can be a function or a tensor
        # lr = 1e-3 * schedule(step)
        # wd = lambda: 1e-4 * schedule(step)
        
        # # PiecewiseConstantDecay also doesn't seem to work properly
        # self.optimizer = AdamW(learning_rate=lr, weight_decay=wd)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
    
    @tf.function
    def train_step(self, makeup_img, non_img):
        with tf.GradientTape() as tape:
            # output
            pred_non = self.model([makeup_img], training=True)
            loss = tf.reduce_mean(tf.square(pred_non-non_img))*100
            
        generator_gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(generator_gradients, self.model.trainable_variables))
        
        return loss
    
    def train(self, train_ds, val_ds):
        best_pnsr = 0.0
        step_counter = 0
        for epoch in range(self.epochs):
            # Train
            step_counter = 0
            total_loss = 0.0
            for makeup_img, non_img in train_ds:
                loss = self.train_step(makeup_img, non_img)
                total_loss = total_loss + loss
                step_counter += 1
                # test
                if step_counter == 100:
                    break
                
            total_loss = total_loss/step_counter
            print('epoch: {}   loss: {}'.format(epoch, total_loss))
            
            pnsr = self.evaluate(epoch, val_ds)     
            if best_pnsr < pnsr:
                best_pnsr = pnsr
                
                # for makeup_img, non_img in val_ds.take(1):
                #     self.generate_images(makeup_img, non_img)

    def evaluate(self, epoch, dataset):  
        psnr_non_mean = 0.0
        count = 0
        for makeup_img, non_img in dataset:
            pred_non = self.model([makeup_img], training=False)
            psnr_non = tf.image.psnr(pred_non, non_img, max_val=1.0)
            __psnr_non_mean = tf.math.reduce_mean(psnr_non)
            psnr_non_mean += __psnr_non_mean
            count += 1
            # test
            if count == 100:
                break
        psnr_non_mean = psnr_non_mean/count
        print('-------- psnr_non: ', psnr_non_mean.numpy(), '----- epoch: ', epoch, '  count: ', count)
        
        return psnr_non_mean

    def generate_images(self, makeup_img, non_img):
        pred_non = self.model([makeup_img], training=False)
        plt.figure(figsize=(15,20))
        display_list = [makeup_img[0], non_img[0], pred_non[0]]
        title = ['Input', 'Non-makeup', 'Predicted']    

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            plt.imshow(display_list[i])
            plt.axis('off')
        #plt.show()


