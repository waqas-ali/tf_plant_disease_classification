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
![](mics/plantvillage-min.png);

Healthy Images in our dataset:
![](mics/healthy_images);

Diseased Images in our dataset:
![](mics/disease_images);

### Data Cleaning and Prepration

We first clean the data so that it can be feed to a machine learning model. 

1-  Dataset consists of RGB color images where each channel has value between 0-255, we first convert the each pixelt value between 0 and 1. 
2-  Each image is of different dimension so to have fix input dimesion we convert the image into 160 x 160 dimension (please note we have tested difference dimensions like 150 x 150, 224 x 224 but got better results on 160 x 160).

3- By following tutoral of [Load images with tf.data](https://www.tensorflow.org/alpha/tutorials/load_data/images#build_a_tfdatadataset), we created our dataset api to feed the data to neural network.
4- We created 3 datasets one for training, one for valid and one for testing. The data distribution is of 75%, 15% and 10% respectivity.
5- For training data we use 64 batch size with shuffle and repeat, whereas for validation the batch size is of 32 and we didn't use shuffle and repeat because there is no need to shuffle the validation data, similarly for test dataset we use batch size of 1 wihtout shuffle and repeat. 

5- ### Model Design and Training



### Use

#### Export your Tensorflow model

In order to serve a Tensorflow model, simply export a SavedModel from your
Tensorflow program.
[SavedModel](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md)
is a language-neutral, recoverable, hermetic serialization format that enables
higher-level systems and tools to produce, consume, and transform TensorFlow
models.

Please refer to [Tensorflow documentation](https://www.tensorflow.org/guide/saved_model#save_and_restore_models)
for detailed instructions on how to export SavedModels.

#### Configure and Use Tensorflow Serving

* [Follow a tutorial on Serving Tensorflow models](tensorflow_serving/g3doc/serving_basic.md)
* Read the [REST API Guide](tensorflow_serving/g3doc/api_rest.md) or [gRPC API definition](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/apis)
* [Use SavedModel Warmup if initial inference requests are slow due to lazy initialization of graph](tensorflow_serving/g3doc/saved_model_warmup.md)
* [Configure models, version and version policy via Serving Config](tensorflow_serving/g3doc/serving_config.md)
* [If encountering issues regarding model signatures, please read the SignatureDef documentation](tensorflow_serving/g3doc/signature_defs.md)

### Extend

Tensorflow Serving's architecture is highly modular. You can use some parts
individually (e.g. batch scheduling) and/or extend it to serve new use cases.

* [Ensure you are familiar with building Tensorflow Serving](tensorflow_serving/g3doc/building_with_docker.md)
* [Learn about Tensorflow Serving's architecture](tensorflow_serving/g3doc/architecture.md)
* [Explore the Tensorflow Serving C++ API reference](https://www.tensorflow.org/tfx/serving/api_docs/cc/)
* [Create a new type of Servable](tensorflow_serving/g3doc/custom_servable.md)
* [Create a custom Source of Servable versions](tensorflow_serving/g3doc/custom_source.md)

## Contribute


**If you'd like to contribute to TensorFlow Serving, be sure to review the
[contribution guidelines](CONTRIBUTING.md).**


## For more information

Please refer to the official [TensorFlow website](http://tensorflow.org) for
more information.
