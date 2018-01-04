# MNIST

This is a SpringBoot web service that takes a jpg or png image of a digit and returns a prediction. It is put together as a Maven project with all the dependencies listed in the POM file. It includes the rest controllers where the image which consumes the image and the Tensorflow library operations which loads a pre-trained model for digit classification. There are currently two models, one for a linear regression model and a Convolutional Neural Network model. Both models were written in python and can be trained on an AWS p2.Large Deep learning image in about a minute.

Run the application as a SpringBoot App and launch it by opening the index.html page. Ex. http://localhost:8080/index.html

On this page you can submit the image to be classified. 
