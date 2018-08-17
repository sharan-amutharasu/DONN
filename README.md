
# DONN: Deep Optimized Nerual Networks

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/sharan-amutharasu/DONN/blob/master/LICENSE)

Machine learning has unarguably been, the biggest contribution of data science to modern science, eliminating the need for human decision making in many processes.

While the field has made giant strides in the recent years, an area of the whole problem that has started consuming increasingly larger resources is the optimization. This involves finding the ideal settings and/or initial conditions for algorithms being used, called parameters and hyper parameters.

The heuristic nature of this process has dampened the extent of automation possible for novice users. But since it is "machine" learning, it is a problem that needs to be addressed.

While there are some efforts being made in this direction, those that deal with deep learning algorithms are even fewer.
Having failed at using these few available options, I decided to build one that made things easier for a project involving deep learning.

And, in case the same could be useful for others, I decided to share a more generic version and this is the resulting product of that aspiration.

If you do use it and run into any problems during the installation or usage, or have some feature requests, please report your problem (only) as an "issue" in this github repository and hopefully, help should be on its way.

Feel free to make your own contributions. Contributions are particularly welcome, in the following areas:
1. Memory and time optimization, specifically in the network training process
2. Support for more network types and layers (specifically, RNN and Convolutional layers)

DONN is built on TensorFlow / Theano / CNTK (neural network backend engines) + Keras (frontend), and is compatible with python 3.

## Installation

Intstalling DONN is simple if you have its dependencies already installed, which are:

1. One of the following neural network backend engines: TensorFlow [Installation link](https://www.tensorflow.org/install/)  / Theano [Installation link](http://deeplearning.net/software/theano/install.html) / CNTK [Installation link](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine)
2. Keras [Installation link](https://keras.io/#installation)

If you have python installed with wheel, open your terminal/command prompt window and enter the following command to install DONN:

```
pip install donn
```

You may need to use a variant of the command based on your version of python and wheel.

Alternatively, if you don't have wheel, you can clone the repository and run the setup file in the base directory using the following commands in your terminal.

```
git clone https://github.com/sharan-amutharasu/donn
cd donn
python setup.py
```

## Usage

DONN is built with ease of usage and novice users in mind and so, can be used with just three lines of code.

```python

optimizer = donn.Optimizer(mode = "classifier")
optimizer.optimize(training_data_x, training_data_y, testing_data_x, testing_data_y)
predictions = donn.predict(prediction_data_x)

```
### Arguments
mode: Mode of operation based on the nature of prediction task. Allowed values are:
    "classifier": If you want to predict discrete labels for datapoints (Eg. Classify whether the data corresponds to a cat or a dog)
    "regressor": If you want to predict continuous values for datapoints (Eg. Predict the weight of the dog described by the data)

training_data_x: Data for training (as numpy array)
training_data_y: Labels for training  (as numpy array or list)
testing_data_x: Data for testing  (as numpy array)
testing_data_y: Labels for testing  (as numpy array or list)

prediction_data_x: Data for prediction (as numpy array)

For advanced users, more control over the process is possible. Here is some detailed documentation for that purpose:
### Algorithm
The optimizer works by calculating increasingly accurate approximations of the ideal parameters. It starts by initially dividing each parameter space into a number of parts, controlled by the user's chosen "level" of optmization complexity. A value from each part is tried along with other such values from other parameter spaces collectively called a combination. 
The best n (again, controlled by user's choice of optimization complexity) combinations are selected and the parameter space is narrowed around each of these combinations.
In successive rounds the parameter space is narrowed down further and further until the best parameters are found (or user defined margins are reached) 

## Documentation
Currently supported types of networks: MLP (Multi Layer Perceptron)
Here is an example with all the possible arguments:
```python

optimizer = donn.Optimizer(mode, name="donn_optimizer", directory=None, layers=None, parameters=None, parameter_precisions=None)
optimizer.optimize(training_data_x, training_data_y, testing_data_x, testing_data_x, validation_data_x=None, validation_data_y=None, loss=None, metric=None, test_metric=None, test_metric_direction=None, verbose=1, max_rounds=2, level=1)
predictions = donn.predict(x_predict, optimizer_name="donn_optimizer", directory=os.getcwd(), probabilities=False, use_one_model=False)

```

### Function: donn.Optimizer():
#### Arguments
mode: Mode of operation based on the nature of prediction task. 
    Allowed values:
        "classifier": If you want to predict discrete labels for datapoints
        "regressor": If you want to predict continuous values for datapoints
name: Name given to optimizer. Files related to the optimizer are stored using this name. If an optimizer already exists in the folder and a new name is not given, it will be overwritten.
    So, it is advisable to give a new name to each optimizer.
    Default value: "donn_optimizer"
directory: Directory where the optimizer files are stored. 
    Default value: The current working directory where the script is executed
layers: The sequence of layers in the neural network.
    Default value: ["input", "activation", "hidden", "activation", "dropout", "hidden", "activation", "dropout", "output"]
    Allowed values: List containing any number of the following strings
        "input" : Input layer into which the data is fed(Dense)
        "hidden" : Fully connected hidden layer of cells (Dense)
        "output" : Output layer from where the label/prediction is received(Dense)
        "activation": Activation layer that scales the output of the previous layer according based on a defined function
        "dropout" : Drops out a defined fraction of the outputs from the previous layer
parameters: Parameters controlling the network
    Default value: {"max_units_for_layers":[100, 100, 100, 1, 1, 100, 1, 1, 100],
                    "activation_function_options": ['relu'],
                    "optimizer_options": ['RMSprop'],
                    "batch_size_range": [128, 128],
                    "max_epochs": 50,
                    "max_dropout_rate": 0.4,
                    "output_activation_function_options": ['sigmoid']
                    }
    Allowed values: The allowed values for each individual parameter are:
        max_units_for_layers: List of integers. List should be the same length as 'layers'. Each item in the list gives the maximum number of cells that the corresponding layer can have.
        activation_function_options: List containing any number of the following strings:
            "relu"
            "softmax"
            "elu"
            "selu"
            "softplus"
            "softsign"
            "tanh"
            "sigmoid"
            "hard_sigmoid"
            "PReLU"
            "LeakyReLU"
            "ThresholdedReLU"
            Or a custom function built using Keras backend
            All the provided functions in the list will be tried and the best one will be chosen
        optmizer_options: List containing any number of the following strings:
            "SGD"
            "RMSprop"
            "Adagrad"
            "Adadelta"
            "Adam"
            "Adamax"
            "Nadam"
            Or a custom optimizer built using Keras backend
            All the provided optimization algorithms in the list will be tried and the best one will be chosen
        batch_size_range: List containing two integers. i.e. the minimum and the maximum batch sizes to be tried. Eg. [64, 512]
        max_epochs: Positive integer. Maximum number of epochs of training to be tried.
        max_dropout_rate: Float between 0 and 1. Maximum Dropout rate to be tried
        output_activation_function_options: List containing any number of the following strings:
            "relu"
            "softmax"
            "elu"
            "selu"
            "softplus"
            "softsign"
            "tanh"
            "sigmoid"
            "hard_sigmoid"
            "PReLU"
            "LeakyReLU"
            "ThresholdedReLU"
            Or a custom function built using Keras backend
            All the provided functions in the list will be tried and the best one will be chosen

parameter_precisions: The minimum margin between the options for parameters that should be tried. i.e. The level of precision expected for each parameter
    Default value: {"precision_for_layers":[5, 5, 5, 1, 1, 5, 1, 1, 5],
                    "precision_batch_size": 8,
                    "precision_epochs": 10,
                    "precision_dropout_rate": 0.1
                    }
    Allowed values: The allowed values for each individual parameter are:
        precision_for_layers: List of integers. List should be the same length as 'layers'. Each item in the list gives the minimum precision for the number of cells that the corresponding layer can have.
        precision_batch_size: Positive integer
        precision_epochs: Positive integer
        precision_dropout_rate: Float between 0 and 1
#### Returns:
Optimizer object (Unoptimized)

### Function: Optimizer.optimize():
#### Arguments
training_data_x: Data for training (as numpy array)
training_data_y: Labels for training  (as numpy array or list)
testing_data_x: Data for testing  (as numpy array)
testing_data_y: Labels for testing  (as numpy array or list)
validation_data_x: Data for validation (as numpy array)
validation_data_y: Labels for validaiton (as numpy array or list)
loss: Loss/Cost function used by the network
    Default values: 
        "mean_absolute_error" (for regression)
        "binary_crossentropy" (for single label classification)
        "categorical_crossentropy" (for multi label classification)
    Allowed values: Any one of the following strings:
        "mean_squared_error"
        "mean_absolute_error"
        "mean_absolute_percentage_error"
        "mean_squared_logarithmic_error"
        "squared_hinge"
        "hinge"
        "categorical_hinge"
        "logcosh"
        "categorical_crossentropy"
        "sparse_categorical_crossentropy"
        "binary_crossentropy"
        "kullback_leibler_divergence"
        "poisson"
        "cosine_proximity"
        Or a custom loss function built using Keras backend
metric: Metric used to evaluate the performance of the network during training and validation
    Default values:
        "binary_accuracy" (for single label classification)
        "categorical_accuracy" (for multi label classification)
        "mae" (for regression)
    Allowed values: Any one of the following strings:
        "mse"
        "mae"
        "categorical_accuracy"
        "sparse_categorical_accuracy"
        "binary_accuracy"
        "top_k_categorical_accuracy"
        "sparse_top_k_categorical_accuracy"
        Or a custom metric built using Keras backend
test_metric: Metric used to evaluate the performance of the network during testing
    Default values:
        sklearn.metrics.accuracy_score (for classification)
        sklearn.metrics.mean_absolute_error (for regression)
    Allowed values:
        Any metric function that accepts true values and predicted values respectively as the first two parameters and returns a integer or float score
test_metric_direction: Whether the test metric score is positively or negatively correlated with performance
    Allowed values:
        "positive" (if higher score is better. Eg. accuracy)
        "negative" (if lower score is better. Eg. error)
verbose: Amount of print statements displayed during the run. 
    Default value: 1
    Allowed values: Integer between 0 and 3
**max_rounds: Maximum number of rounds of optimization to be carried out. The higher the value, the better the results and higher the resource consumption
    Default value: 5
    Allowed values: Positive integer
level: Degree of optimization complexity. The higher the value, the better the results and higher the resource consumption
    Default value: 2
    Allowed values: Positive integer **

#### Returns:
Optimizer object (optimized)

### Funciton: donn.predict():
#### Arguments
x_predict: Data for prediction (as numpy array)
optimizer_name: Name of the optimizer to use for prediction.
    Default value: "donn_optimizer"
    Allowed values: String
directory: Directory in which the optimizer data is present
    Default value: Current working directory
    Allowed values: Any legal path in the system
probabilities: If true, in classifier mode, instead of just returning the predicitons, the function returns two objects: (1) The classes (2) The predicted probabilities
    Default value: False
    Allowed values: True, False
use_one_model: If true, instead of using multiple best models and combining their results, prediction is made using only the one best model
    Default value: False
    Allowed values: True, False

#### Returns:
Predictions: Numpy array of same length as 'x_predict' (If probabilities=False)
Classes, Prediction_probabilites: List of class names, Numpy array of same length as 'x_predict' (If probabilities=False)


That should be plenty to get started with DONN.
Enjoy!

## Contributors
Sharan Amutharasu - Author
