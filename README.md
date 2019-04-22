# Plant Classification in Healthy and Disease

----

## Problem Statement & Introduction

In our work we target the Indoor farming. Indoor farming is a method of growing crops or plants, entirely indoors. This method of farming often implements growing methods such as hydroponics and utilizes artificial lights to provide plants with the nutrients and light levels required for growth. A wide variety of plants can be grown indoors, but fruits, vegetables, and herbs are the most popular. Some of the most popular plants grown indoors are usually crop plants like lettuce, tomatoes, peppers, and herbs.

When growing indoors, many indoor farmers appreciate having more control over the environment than they do when they are using traditional farming methods. Light amounts, nutrition levels, and moisture levels can all be controlled by the farmer when they are growing crops solely indoors. They also need a system that monitor the health of a plant and generate alerts if a plant got a disease. The system need to be automatic which captures the real time images of plants at various angels and then classify it with either healthy or diease and provide a mitigation to redeuce the disease effect.

Keep this thing in mind we have gathered the data of different plants of both category healthy and disease. I would like to thanks Pam Loreto which provides the data of these categories. 

The data consists of following plants:

-   Basil
-   Kale
-   Lettuce
-   Mint
-   Coriander
-   Parsley

## Solution

We developed a machine learning model in Tensorflow 2.0 using keras API which classify the plant image into either healthy or disease. The problem is a binary problem in which we have to calssify an image either healthy (1) or diseased (0).


## Technicallity

### Data exploration


## Documentation

### Set up

The easiest and most straight-forward way of using TensorFlow Serving is with
Docker images. We highly recommend this route unless you have specific needs
that are not addressed by running in a container.

*   [Install Tensorflow Serving using Docker](tensorflow_serving/g3doc/docker.md)
    *(Recommended)*
*   [Install Tensorflow Serving without Docker](tensorflow_serving/g3doc/setup.md)
    *(Not Recommended)*
*   [Build Tensorflow Serving from Source with Docker](tensorflow_serving/g3doc/building_with_docker.md)
*   [Deploy Tensorflow Serving on Kubernetes](tensorflow_serving/g3doc/serving_kubernetes.md)

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
