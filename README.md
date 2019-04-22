# Plant Classification in Healthy and Disease

----

## Problem Statement & Introduction

In our work we target the Indoor farming. Indoor farming is a method of growing crops or plants, entirely indoors. This method of farming often implements growing methods such as hydroponics and utilizes artificial lights to provide plants with the nutrients and light levels required for growth. A wide variety of plants can be grown indoors, but fruits, vegetables, and herbs are the most popular. Some of the most popular plants grown indoors are usually crop plants like lettuce, tomatoes, peppers, and herbs.

When growing indoors, many indoor farmers appreciate having more control over the environment than they do when they are using traditional farming methods. Light amounts, nutrition levels, and moisture levels can all be controlled by the farmer when they are growing crops solely indoors. They also need a system that monitor the health of a plant and generate alerts if a plant got a disease. The system need to be automatic which captures the real time images of plants at various angels and then classify it with either healthy or diease and provide a mitigation to redeuce the disease effect.

Keep this thing in mind we have gathered the data of different plants of both category healthy and disease. I would like to thanks Pam Loreto which provides the data of these categories. 


## Solution

We developed a machine learning model in Tensorflow 2.0 using keras API which classify the plant image into either healthy or disease. The problem is a binary problem in which we have to calssify an image either healthy (1) or diseased (0).


## Technicality

### Data exploration

The data consists of following plants:

-   Basil
-   Kale
-   Lettuce
-   Mint
-   Coriander
-   Parsley

There are total 3374 images in the data set in which 1943 images are of cateogry diseased  and 1431 images are of category healthy. The size of each image is different so the image dimension. Most of the images are in jpeg but also contains some images in .png and gif.

There is another dataset for plant disease classification [PlantVillage Disease Classification Challenge](https://www.crowdai.org/challenges/plantvillage-disease-classification-challenge). The main difference between PlantVillage and our dataset is that we have riched images unlike PlantVillage in which there are only images of leaves not whole plants. In real world it is very difficult and costly for the system to capture the images of each leaf seperately and classify them into healthy and diseased.  

You can see some examples of both datasets. 

PlantVillage
![](misc/plantvillage-min.png)

Healthy Images in our dataset:
![](misc/healthy_images.png)

Diseased Images in our dataset:
![](misc/disease_images.png)

### Data Cleaning and Prepration

We first clean the data so that it can be feed to a machine learning model. 

-   Dataset consists of RGB color images where each channel has value between 0-255, we first convert the each pixelt value between 0 and 1. 
-   Each image is of different dimension so to have fix input dimesion we convert the image into 160 x 160 dimension (please note we have tested difference dimensions like 150 x 150, 224 x 224 but got better results on 160 x 160).
-   we follow tutoral of [Load images with tf.data](https://www.tensorflow.org/alpha/tutorials/load_data/images#build_a_tfdatadataset), to creat  dataset api for feeding the data to neural network.
-   We created 3 datasets one for training, one for valid and one for testing. The data distribution is of 75%, 15% and 10% respectivity.
-   For training data we use 64 batch size with shuffle and repeat, whereas for validation the batch size is of 32 and we didn't use shuffle and repeat because there is no need to shuffle the validation data, similarly for test dataset we use batch size of 1 wihtout shuffle and repeat. 

### Model Design and Training

we have tested different architecture of convolution neural network with different hyper parameters. We acheived best results on Vgg16 pretained model with fune tunning on last two convolutional layers. Below is the architecutre of our model

-   We use Vgg16 model, wihtout top included, pretained on imagenet followed by with two dense layers each having 512 units with Dropout of 50% between them.
-   Each dense layer has relu activation function follow by 1 output dense layer with sigmoid activation function.
-   For loss we use binary_crossentropy and for optimizer we use Adam with learning rate 0.00001
-   For model callback we use early stop to save only best model which have lowest validation loss and Reduce learning rate if there is no variation in the validation loss.
-   The model was trained on train dataset and after each epcoh the model is tested against validation dataset. 
-   Model only trained the parameter of last 2 dense layer and convolutional layer parameter was kept freeze.
-   Once model is trained we finetuned the model by setting last 2 convolutional layer trainable and retrained the model on training data. 
-   In fine tuning we used lower value of learning rate as compare to first training.

Below is the screenshot of code snippet of model design

![](misc/vgg16.png)

We have also tested different architecutre design with different parameters, some are mentioned below

-   Simple convolutional neural network with 3 Conv2D layer with 2 Dense layer.
-   A Separable convolution neural network with 4 SeparableConv2D follow by 2 Dense layer.
-   Pretained version of Vgg16, Vgg19, inception, Xception, mobileNet, DenseCNN and ResNet on imagenet dataset and also on random intialization. We have seen that on DenseCNN and ResNet the model perform very badly where as on Inception and Xception we receive fair accuracy. On MobileNet in pretained version the model had bad accuracy (less than 70%) but on random initialization of weight with trainable True model perform good as compare to Inception and Xception.

### Validation and Test Loss/Accuracy

- 



## Contribute

**If you'd like to contribute to TensorFlow Serving, be sure to review the
[contribution guidelines](CONTRIBUTING.md).**


## For more information

Please refer to the official [TensorFlow website](http://tensorflow.org) for
more information.
