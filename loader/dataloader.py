import os
import numpy as np

import tensorflow as tf

#from loader.aug import ImgAugTransform


class Dataloader():
    def __init__(self, config):
         self.img_path = config['training']['img_path']
         self.batch_size = config['training']['batch_size']


    def loader(self):
        train_data_list = self.prep_data(str(self.img_path + '/train/makeup/'))
        val_data_list = self.prep_data(str(self.img_path + '/val/makeup/'))
        test_data_list = self.prep_data(str(self.img_path + '/test/makeup/'))

        np.random.shuffle(train_data_list)
        np.random.shuffle(val_data_list)
        np.random.shuffle(test_data_list)

        train_dataset = tf.data.Dataset.from_tensor_slices(train_data_list)
        train_dataset = train_dataset.map(self.load_image_train,
                                        num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.shuffle(self.batch_size)
        train_dataset = train_dataset.batch(self.batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices(val_data_list)
        val_dataset = val_dataset.map(self.load_image_val)
        val_dataset = val_dataset.batch(self.batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices(test_data_list)
        test_dataset = test_dataset.map(self.load_image_val)
        test_dataset = test_dataset.batch(self.batch_size)

        return train_dataset, val_dataset, test_dataset


    def load(self, image_file):
        makeup_img_file, non_img_file =  tf.split(image_file, 2)
        
        makeup_img = tf.io.read_file(makeup_img_file[0])
        makeup_img = tf.image.decode_jpeg(makeup_img, channels=3)
        
        non_img = tf.io.read_file(non_img_file[0])
        non_img = tf.image.decode_jpeg(non_img, channels=3)

        # Convert both images to float32 tensors
        makeup_img  = tf.cast(makeup_img, tf.float32)
        non_img = tf.cast(non_img, tf.float32)
        
        return makeup_img, non_img


    def processing_image(self, makeup_img, non_img):
        makeup_img = (makeup_img / 255.0)
        non_img = (non_img / 255.0)
        
        return makeup_img, non_img
    
    
    def random_flip(self, makeup_img, non_img):
        if tf.random.uniform(()) > 0.5:
            # Random mirroring
            makeup_img = tf.image.flip_left_right(makeup_img)
            non_img = tf.image.flip_left_right(non_img)

        return makeup_img, non_img


    def load_image_train(self, image_file):
        makeup_img, non_img = self.load(image_file)
        makeup_img, non_img = self.random_flip(makeup_img, non_img)
        makeup_img, non_img = self.processing_image(makeup_img, non_img)

        return makeup_img, non_img


    def load_image_val(self, image_file):
        makeup_img, non_img = self.load(image_file)
        makeup_img, non_img = self.processing_image(makeup_img, non_img)

        return makeup_img, non_img 

    def prep_data(self, path):
        makeup_img_list = [os.path.join(path, f) for f in os.listdir(path)]
        data_list = [[i, i.replace('makeup','non-makeup')] for i in makeup_img_list]
        return data_list

