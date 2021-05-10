# Neural Network from scratch for digit classification

* Prepare the data.
* Create a loss function and a metric.
* Select a basic model.
* Create a class to be able to train the model using the loss function and showing the metric results.
* Compare the results with a more complex model.

## Introduction
In this notebook I used the foundations behind Neural Networks to develop a model able to classify **handwritten digits**.

## Data

Consists of 70000 black and white images of 28\*28 pixels with a label associated to each image (0-9 floating point number). Each pixel is an integer between 0 and 255 (both included), so I scaled them between 0 and 1 for better performance of the model. Also, I converted each label to an integer. The result are 70000 numpy arrays of 784 floating point numbers of range 0-1, with an integer between 0 to 9 as the label associated with each.

Then I randomly selected 15% of the images (10500) to use them as validation set.

The basic operation of a neural network is matrix multiplication of the input data by a weight matrix, the parameters of this weight matrix are optimized by calculating their gradients based in a loss function, so in order to be able to do this I converted each image (a numpy array of 28\*28 numbers) to a pytorch tensor, which is similar to a numpy array but keeps track of the gradients of the parameters.

Lastly, to make the computation faster I divided the data into mini-batchs of size 256 to be able to pass them at the same time to the GPU.

## Loss Function and Metric
As a loss function I use **Cross Entropy**. It consists of log_softmax + negative log likelihood.

As a metric, because this is a classification task, I use **Accuracy** which seems to me as the most interpretable.

## Model

### First model
As a first model I use a very basic linear model which consists in a matrix multiplication plus a bias: **x@weights + bias**. Being **x** each mini-batch of size 256x784, **@** the symbol used for matrix multiplication in Python, **weights** the parameters matrix that I multiply the images by and **bias** the parameter that I add after the multiplication.

The **weights** matrix is of size 784x10, because I want and output for each label.

### Second model
For the second model  I add an activation function (Rectified Linear Unit) and a second layer. So it is a matrix multiplication, followed by a reLU and another matrix multiplication.

In this notebook I use some basic pytorch functions and classes (F.cross_entropy, nn.Linear) to develop a deep learning model from scratch able to classify digits with 97% accuracy on validation set.
Also, I create my own implementation of fastai Learner class which allowed me to train the model much faster.

# Learner Class
To be able to use all this pieces as a whole I create a class called learner2. Each instance of this class have a dataloaders object, a model, an optimation function , a loss function and a metric. It has two basic funtionalities:
* **.fit**: to be able to train the model with a specific learning rate.
* **.predict**
## Conclusion
