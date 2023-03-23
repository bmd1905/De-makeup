# De-makeup

>Onging Project

This project aims to create a neural network model that can automatically remove makeup from facial images. It is built using UNet with ResNet34 as backbone and trained on a custom dataset of facial images with and without makeup.

<p align="center">
  <img src="https://user-images.githubusercontent.com/90423581/221457342-94f20315-b2bd-450b-b2ce-b27aa88ab62e.png" data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width="700" height="300" />
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/90423581/221457534-53cb45ff-4625-4e42-8b1b-b62b8a0ab11f.png" data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width="700" height="300" />
</p>


# Overview
De-makeup using UNet with ResNet34 as backbone is a project that uses deep learning techniques to automatically remove makeup from facial images. The model is trained on a custom dataset of facial images with and without makeup, and the output is a de-makeup image that accurately represents the user's natural appearance. This project has many potential applications in the beauty industry and beyond, including personal photo albums, social media, and online beauty stores. By making it easier for individuals to see themselves without makeup and providing more accurate product depictions, De-makeup can help promote more authentic beauty standards.

# Architecture
## Encoder
The encoder of the model based on ResNet34, before going to my customed ResNet34, let look at the vanilla ResNet34.

ResNet34 is a deep convolutional neural network architecture that belongs to a family of models known as ResNets. It consists of 34 layers and uses residual connections to overcome the problem of vanishing gradients in very deep neural networks. ResNet34 has been widely used for image classification, object detection, and segmentation, and has demonstrated state-of-the-art performance on many benchmark datasets. Its success has contributed to the development of deeper and more complex neural network architectures and remains a popular and powerful model in the field of deep learning.

In my case, I replaced **Batch Normalization** and **ReLU** activation function by **Instance Normalization** and **GeLU activation functiton** respectively.
## Decoder
The decoder of model is simply Conv2DTranspose, skip connection with encoder, and Conv2D. The first one I used was Conv2DTranspose to upsample the feature maps, followed by a skip connection with the encoder, and the last one is Conv2D. I repeated this block 5 times to change the latent space from ```7x7x512``` to ```224x224x3``` (the same with the input image).

# How to use my model
First, you need to clone this project:
```
git clone https://github.com/bmd1905/De-makeup
```
To run the project, you'll need to install the following dependencies:
```
pip install -r requirements.txt
```

You can then use the predict.py script to apply the de-makeup model to an image. For example:
```
python predict.py --image path/to/image.jpg --weigths path/to/weights.h5
```
Alternative way, using my [jupyter notebook](https://github.com/bmd1905/De-makeup/blob/main/predict.ipynb) to follow and play around with.

# Training
If you want to train from scratch or fine-tune the model on your own custom dataset, you need to modify the [config.yml](https://github.com/bmd1905/De-makeup/blob/main/config.yml) file according to your needs. If you want to customize the model architecture, you can modify the [resnet34_unet.py](https://github.com/bmd1905/De-makeup/blob/main/model/resnet34_unet.py) file. Once you've finished configuring, all you need to do is run the following script.

```
python train.py
```


# References
* AIO-2022 Course
