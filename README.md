# De-makeup

### Ongoing Project
This project aims to create a neural network model that can automatically remove makeup from facial images. It is built using UNet with ResNet34 as backbone and trained on a custom dataset of facial images with and without makeup.

<p align="center">
  <img src="https://user-images.githubusercontent.com/90423581/218090114-ce8a6421-e5c3-48f8-a677-a9183e7ec0f2.png" data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width="750" height="400" />
</p>

# Overview
De-makeup using UNet with ResNet34 as backbone is a project that uses deep learning techniques to automatically remove makeup from facial images. The model is trained on a custom dataset of facial images with and without makeup, and the output is a de-makeup image that accurately represents the user's natural appearance. This project has many potential applications in the beauty industry and beyond, including personal photo albums, social media, and online beauty stores. By making it easier for individuals to see themselves without makeup and providing more accurate product depictions, De-makeup can help promote more authentic beauty standards.

# Getting Started
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


<p align="center">
  <img src="https://user-images.githubusercontent.com/90423581/219059622-f1ec82c4-7dc1-4ddb-bd32-dc3696ff7e66.png" data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width="750" height="400" />
</p>


# References
* AIO-2022 Course
