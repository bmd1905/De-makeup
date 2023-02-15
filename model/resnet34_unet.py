import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow_addons as tfa


class ReNet34_UNet():
    def __init__(self, input_shape):
        super(ReNet34_UNet, self).__init__()
        self.input_shape = input_shape

        self.model = self.build_model()

        #return self.build_model()

    
    def build_model(self):
        x_input = tf.keras.layers.Input(self.input_shape)

        # Create encoder
        ec, skips = self.encoder(x_input)

        # Create decoder
        input_shape_dc = (7, 7, 512)
        dc = self.decoder(ec, skips)

        model = tf.keras.models.Model(inputs=x_input, outputs = dc, name = "ResNet34_Unet")

        return model


    def encoder(self, x_input):
        skips = []
        #x_input = tf.keras.layers.Input(self.input_shape)
        #x = tf.keras.layers.ZeroPadding2D((3, 3))(x_input)
        # Step 2 (Initial Conv layer along with maxPool)
        x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(x_input)
        x = tfa.layers.InstanceNormalization()(x)
        x = tf.keras.layers.Activation('gelu')(x)
        skips.append(x)
        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
        # Define size of sub-blocks and initial filter size
        block_layers = [3, 4, 6, 3]

        filter_size = 64
        # Step 3 Add the Resnet Blocks
        for i in range(4):
            if i == 0:
                # For sub-block 1 Residual/Convolutional block not needed
                for j in range(block_layers[i]):
                    x = self.identity_block(x, filter_size)
            else:
                # One Residual/Convolutional Block followed by Identity blocks
                # The filter size will go on increasing by a factor of 2
                filter_size = filter_size*2
                x = self.convolutional_block(x, filter_size)
                for j in range(block_layers[i] - 1):
                    x = self.identity_block(x, filter_size)
            
            skips.append(x)

        # Because of the last added skip is 7x7x512
        skip_layer_input = skips.pop()

        return x, skips


    def decoder(self, ec, skips=None):
        x = tf.keras.layers.Conv2DTranspose(256, (2,2),strides=(2,2),padding='same')(ec)
        skip_layer_input = skips.pop()
        merge = tf.keras.layers.concatenate([x, skip_layer_input], axis=-1)
        x = tf.keras.layers.Conv2D(256, 3,  activation='gelu',padding='same')(merge)


        x = tf.keras.layers.Conv2DTranspose(128, (2,2),strides=(2,2),padding='same')(x)
        skip_layer_input = skips.pop()
        merge = tf.keras.layers.concatenate([x, skip_layer_input], axis=-1)
        x = tf.keras.layers.Conv2D(128,3, activation='gelu',padding='same')(merge)


        x = tf.keras.layers.Conv2DTranspose(64, (3,3),strides=(2,2),padding='same')(x)
        skip_layer_input = skips.pop()
        merge = tf.keras.layers.concatenate([x, skip_layer_input], axis=-1)
        x = tf.keras.layers.Conv2D(64,3, activation='gelu',padding='same')(merge)

        x = tf.keras.layers.Conv2DTranspose(64, (2,2),strides=(2,2),padding='same')(x)
        skip_layer_input = skips.pop()
        merge = tf.keras.layers.concatenate([x, skip_layer_input], axis=-1)
        x = tf.keras.layers.Conv2D(64,3, activation='gelu',padding='same')(merge)

        x = tf.keras.layers.Conv2DTranspose(3,(2,2),strides=(2,2),padding='same')(x)
        x = tf.keras.layers.Conv2D(3,3, activation='gelu',padding='same')(x)

        return x


    def identity_block(self, x, filter):
        # copy tensor to variable called x_skip
        x_skip = x
        # Layer 1
        x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = tf.keras.layers.Activation('gelu')(x)
        # Layer 2
        x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
        x = tfa.layers.InstanceNormalization()(x)
        # Add Residue
        x = tf.keras.layers.Add()([x, x_skip])     
        x = tf.keras.layers.Activation('gelu')(x)

        return x

    def convolutional_block(self, x, filter):
        # copy tensor to variable called x_skip
        x_skip = x
        # Layer 1
        x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same', strides = (2,2))(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = tf.keras.layers.Activation('gelu')(x)
        # Layer 2
        x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
        x = tfa.layers.InstanceNormalization()(x)
        # Processing Residue with conv(1,1)
        x_skip = tf.keras.layers.Conv2D(filter, (1,1), strides = (2,2))(x_skip)
        # Add Residue
        x = tf.keras.layers.Add()([x, x_skip])     
        x = tf.keras.layers.Activation('gelu')(x)

        return x