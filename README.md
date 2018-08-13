
# DONN: Deep Optimized Nerual Networks

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/keras-team/keras/blob/master/LICENSE)

Machine learning is the unarguably the biggest contribution of data science to science, eliminating the need for human decision making in many processes

While the field has made giant strides in the recent years, an area of the whole problem that has started consuming increasingly larger resources is the optimization. This involves finding the ideal settings or initial conditions for algorithm being used, called parameters and hyper parameters.

The heuristic nature of this process has dampened the extent of automation possible for novice users. But since it is "machine" learning, it is a problem that needs to be addressed.

While there are some efforts being made in this direction, those that deal with deep learning algorithms are even fewer.
Having failed at using these few available options, I decided to build one that makes things easier for my specific application.

And, in case the same could be useful for others, I decided to share a more generic version and this is the resulting product of that aspiration.

If you do use it and run into any problems during the installation or usage, or have some feature requests, please report your problem (only) as an "issue" in this github repository and hopefully, help should be on its way.

Feel free to make your own contributions. Contributions are particularly welcome, in the following areas:
1. Memory and time optimization, specifically in the network training process
2. Support for more network types and layers

DONN is built on TensorFlow / Theano / CNTK (neural network backend engines) + Keras (frontend), and is compatible with python 3.

## Installation

Intstalling DONN is simple if you have its dependencies already installed, which are:

1. A neural network backend engine: TensorFlow / Theano / CNTK
2. Python + Wheel + Keras 

If you have python and wheel installed, open your terminal/command prompt window and enter the following command to install DONN:

```
pip install donn
```

You may need to use a variant of the command based on your version of python and wheel.

Alternatively, if you don't have wheel, you can clone the repository and run the setup file in the base directory.

```
git clone https://github......
cd donn
python setup.py
```

## Usage

DONN is built with ease of usage in mind and so, can be used with just three lines of code.

```python

optimizer = donn.Optimizer(mode = "classifier")
optimizer.optimize(training_data_x, training_data_y, testing_data_x, testing_data_y)
predictions = donn.predict(prediction_data_x)

```
### Arguments
mode: 

training_data_x: 
training_data_y: 
testing_data_x: 
testing_data_y:

prediction_data_x: 

If you want more control over the process please look at the following version showing all the parameters, you can use for additional control.

```python

optimizer = donn.Optimizer(mode = "classifier")
optimizer.optimize(training_data_x, training_data_y, testing_data_x, testing_data_y)
predictions = donn.predict(prediction_data_x)

```
### Arguments

That should be plenty for you to get started with DONN.
Enjoy!

## Contributors
Sharan Amutharasu - Author