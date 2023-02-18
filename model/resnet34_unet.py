import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Conv2DTranspose
import tensorflow_addons as tfa


class ReNet34_UNet(Model):
    def __init__(self, input_shape):
        super(ReNet34_UNet, self).__init__()
        self.input_shape_model = input_shape

        #self.model = self.build_model()
        inputs = tf.keras.layers.Input(self.input_shape_model)
        
        encoder, self.skips = self.build_encoder(inputs)
        decoder = self.build_decoder(encoder)

        self.model = tf.keras.models.Model(inputs=inputs, outputs = decoder, name = "ResNet34_Unet")

    def __call__(self, makeup_imgs, training=True):
        return self.model(makeup_imgs, training)
        # encoded = self.encoder(makeup_imgs)
        # decoded = self.decoder(encoded)
        # return decoded
    
    # def __call__(self, makeup_img, training=True):
    #     return self.model([makeup_img], training=training)

    
    # def build_model(self):
    #     x_input = tf.keras.layers.Input(self.input_shape_model)

    #     # Create encoder
    #     ec, skips = self.encoder(x_input)

    #     # Create decoder
    #     input_shape_dc = (7, 7, 512)
    #     dc = self.decoder(ec, skips)

    #     model = tf.keras.models.Model(inputs=x_input, outputs = dc, name = "ResNet34_Unet")

    #     return model


    def build_encoder(self, inputs):
        
        skips = []
        #x_input = tf.keras.layers.Input(self.input_shape)
        #x = tf.keras.layers.ZeroPadding2D((3, 3))(x_input)
        # Step 2 (Initial Conv layer along with maxPool)
        x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
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

        #encoder = Model(encoder_input, x)

        return x, skips


    def build_decoder(self, encoder):
        skips = self.skips
        #decoder_input = tf.keras.layers.Input(self.encoder.layers[-1].output_shape[1:])
        #decoder_input = tf.keras.layers.Input((7, 7, 512))

        x = tf.keras.layers.Conv2DTranspose(256, (2,2),strides=(2,2),padding='same')(encoder)
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

        #decoder = Model(self., x)

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


# import tensorflow as tf
# from tensorflow.keras.layers import *
# from tensorflow.keras.models import Model
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
# from tensorflow.keras.models import Model



# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Conv2DTranspose
# from tensorflow.keras.applications import ResNet50

# class ReNet34_UNet(Model):
#     def __init__(self, input_shape):
#         super(ReNet34_UNet, self).__init__()
        
#         self.encoder = ResNet50(include_top=False, input_shape=input_shape)
        
#         self.decoder = self.build_decoder()

#     def build_decoder(self):
#         decoder_input = Input(self.encoder.layers[-1].output_shape[1:])
        
#         x = decoder_input
#         for layer in self.encoder.layers[-2::-1]:
#             if isinstance(layer, MaxPooling2D):
#                 x = UpSampling2D()(x)
#             elif isinstance(layer, Conv2D):
#                 x = Conv2DTranspose(layer.filters, layer.kernel_size, strides=layer.strides, padding='same')(x)
#             else:
#                 continue
#             x = Concatenate()([x, layer.output])
        
#         decoder_output = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)
        
#         decoder = Model(decoder_input, decoder_output)
#         return decoder
    
#     def call(self, inputs):
#         encoded = self.encoder(inputs)
#         decoded = self.decoder(encoded)
#         return decoded

