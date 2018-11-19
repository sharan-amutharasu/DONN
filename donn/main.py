
# coding: utf-8

# In[ ]:


import tools
import os
import datetime
import itertools as it
import gc
import pickle
import numpy as np
import layers


# In[2]:


from keras.constraints import maxnorm
from keras.layers import Activation, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU, PReLU, ThresholdedReLU, ELU
from keras.models import Sequential
from keras.models import load_model as keras_load_model
from keras import regularizers, optimizers, metrics
# from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from keras import backend as K
from keras.utils import np_utils


# In[3]:


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error


# In[4]:


allowed_layers = ["input", "hidden", "activation", "output"]


# In[5]:


class Optimizer(object):
    """Main class that does all the work.
    Initializes new or old instance of an optimizer, 
    Runs all the possible combinations of parameters,
    Identifies & stores the optimal one(s),
    Predicts output based on optimal model(s)
    # Arguments
        mode: Nature of task: "classification" or "regression"
        name: name given to the instance of the optimizer. 
            Defaults to "donn_optimizer". Further instances 
            will overwrite the data and models of previous 
            instance unless a different name is specified.
        directory: The directory where you want the data and 
            the models to be stored. Defaults to the directory 
            from where the script is run
    """
    
    # Initialize instance
    def __init__(self, mode, name="donn_optimizer", directory=os.getcwd(), layers=None, parameters=None, parameter_precisions=None):
        
        # If data does not exist for the current instance name, create it
        self.data_filename = str(name + "-data.pickle")
        try:
            self.data = tools.read_data(directory, self.data_filename)
            print("Optimizer found with given name. Loading the old optimizer")
        except FileNotFoundError:
            print("No old optimizer data found for given name. Initiating new optimizer")
            self.data = {"stage":0}
            
        if self.data["stage"] == 0:
            self.data = {"combs":{}, "combs_comp":{}, "best":{"best":{}}, "grids":{}, "stage":0}
            self.data["optimized"] = False
            self.data["name"] = name
            self.data["directory"] = directory
            
            
            ## Check the provided layers
            if layers is not None:
                if type(layers) != list:
                    raise TypeError("layers should be of type 'list'")
                if len(layers) < 2:
                    raise ValueError("Minimum of two layers must be present")
                for i in range(0,len(layers)):
                    if layers[i] not in allowed_layers:
                        raise ValueError("Unrecognised type of layer: %s " % layer)
                    if i == layers[len(layers)-1] and layers[i] != "output":
                        raise ValueError("The last layer must be 'output'. It cannot be %s " % layers[len(layers)-1])
                    if i != layers[len(layers)-1] and layers[i] == "output":
                        raise ValueError("Only the last layer can be 'output'")
                self.data["layers"] = layers
            else:
                self.data["layers"] = ["input", "activation", "hidden", "activation", "hidden", "activation", "output"]
            
            ## check and store parameters
            max_units_for_layers = []
            for layer in self.data["layers"]:
                max_units_for_layers.append(self.get_default_values(layer, "range")[1])
            default_params = {"max_units_for_layers":max_units_for_layers,
                              "activation_function_options": self.get_default_values("activation_function", "range"),
                              "optimizer_options": self.get_default_values("optimizer", "range"),
                              "batch_size_range": self.get_default_values("batch_size", "range"),
                              "max_epochs": self.get_default_values("epochs", "range")[1],
                              "max_dropout_rate": self.get_default_values("dropout_rate", "range")[1],
                              "output_activation_function_options": self.get_default_values("output_activation_function", "range")
                             }
            
            if parameters is not None:
                self.data["parameters"] = {}
                for param in parameters.keys():
                    if param in default_params:
                        self.data["parameters"][param] = parameters[param]
                        if param == "max_units_for_layers":
                            if len(parameters[param]) != len(self.data["layers"]):
                                raise ValueError("Length of 'layers' provided and length of 'parameters' option 'max_units_for_layers' should be the same")
                    else:
                        raise ValueError("Unrecognised parameter in 'parameters': %s " % param)
                for param in default_params.keys():
                    if param not in parameters:
                        self.data["parameters"][param] = default_params[param]
            else:
                self.data["parameters"] = default_params
            
            ## Check and store parameter_precisions 
            precision_for_layers = []
            for layer in self.data["layers"]:
                precision_for_layers.append(self.get_default_values(layer, "min"))
            default_param_precisions = {"precision_for_layers":precision_for_layers,
                                        "precision_batch_size": self.get_default_values("batch_size", "min"),
                                        "precision_epochs": self.get_default_values("epochs", "min"),
                                        "precision_dropout_rate": self.get_default_values("dropout_rate", "min")
                                       }
            
            if parameter_precisions is not None:
                self.data["parameter_precisions"] = {}
                for param in parameter_precisions.keys():
                    if param in default_param_precisions:
                        self.data["parameter_precisions"][param] = parameter_precisions[param]
                        if param == "precision_for_layers":
                            if len(parameter_precisions[param]) != len(self.data["layers"]):
                                raise ValueError("Length of 'layers' provided and length of 'parameters' option 'max_units_for_layers' should be the same")
                    else:
                        raise ValueError("Unrecognised parameter in 'parameter_precisions': %s " % param)
                for param in default_param_precisions.keys():
                    if param not in parameter_precisions:
                        self.data["parameter_precisions"][param] = default_param_precisions[param]
            else:
                self.data["parameter_precisions"] = default_param_precisions
            
            
            ## Create base range for first round of optimization
            self.data["base_range"] = {}
            for i in range(0, len(self.data["layers"])):
                if i == 0:
                    count = 0
                else:
                    count = self.data["layers"][:i].count(self.data["layers"][i])
                key_string = str(self.data["layers"][i] + "_layer_" + str(count+1) + "_units")
                self.data["base_range"][key_string] = {"range":[1,self.data["parameters"]["max_units_for_layers"][i]],
                                                       "min":self.data["parameter_precisions"]["precision_for_layers"][i]
                                                      }
                
            self.data["base_range"]["activation_function"] = {"range":self.data["parameters"]["activation_function_options"]}
            self.data["base_range"]["optimizer"] = {"range":self.data["parameters"]["optimizer_options"]}
            self.data["base_range"]["batch_size"] = {"range":self.data["parameters"]["batch_size_range"],
                                                     "min":self.data["parameter_precisions"]["precision_batch_size"]
                                                    }
            self.data["base_range"]["epochs"] = {"range":[2, self.data["parameters"]["max_epochs"]],
                                                 "min":self.data["parameter_precisions"]["precision_epochs"]
                                                }
            self.data["base_range"]["dropout_rate"] = {"range":[0, self.data["parameters"]["max_dropout_rate"]],
                                                       "min":self.data["parameter_precisions"]["precision_dropout_rate"]
                                                      }
            self.data["base_range"]["output_activation_function"] = {"range":self.data["parameters"]["output_activation_function_options"]}
            
                            
            
            ## Production
#             self.data['base_range'] = {"input_layer_units":{"range":[1,500], "min":1},
#                                "hidden_layer_1_units":{"range":[1,500], "min":1},
#                                "hidden_layer_2_units":{"range":[1,500], "min":1},
#                                "hidden_layer_3_units":{"range":[1,500], "min":1},
#                                "activation":{"range":['tanh', 'elu']},
#                                "optimizer":{"range":['RMSprop', 'Adam']},
#                                "batch_size":{"range":[64], "min":8},
#                                "epochs":{"range":[10], "min":2},
#                                "dropout_rate":{"range":[0], "min":0.05}
#                               }
            if mode.lower() == "regressor":
                self.data["mode"] = 'regressor'
            elif mode.lower() == "classifier":
                self.data["mode"] = 'classifier'
            else:
                raise ValueError('Invalid value for "mode". Please pass in either "regressor" or "classifier". ')
            self.data["stage"] = 1
        
        ## Save data to local file
        tools.save_data(self.data, self.data["directory"], self.data_filename)
        print("Optimizer created")
    
#     def get_default_values(self, param, typ):
#         if param == "input" or param == "hidden" or param == "output":
#             if typ == "range":
#                 return [1,100]
#             if typ == "min":
#                 return 1
#         elif param == "activation" or param == "dropout":
#             if typ == "range":
#                 return [1,1]
#             if typ == "min":
#                 return 1
#         elif param == "activation_function":
#             if typ == "range":
#                 return ['sigmoid', 'softmax', 'relu']
#             else:
#                 return None
#         elif param == "optimizer":
#             if typ == "range":
#                 return ['SGD', 'RMSprop', 'Adagrad']
#             else:
#                 return None
#         elif param == "batch_size":
#             if typ == "range":
#                 return [64, 256]
#             if typ == "min":
#                 return 8
#         elif param == "epochs":
#             if typ == "range":
#                 return [5, 100]
#             if typ == "min":
#                 return 2
#         elif param == "dropout_rate":
#             if typ == "range":
#                 return [0, 0.4]
#             if typ == "min":
#                 return 0.1
#         elif param == "output_activation_function":
#             if typ == "range":
#                 return ['sigmoid']
#             else:
#                 return None
#         else:
#             raise ValueError("Unrecongised parameter: %s" % param)
    
    ## Testing
    def get_default_values(self, param, typ):
        if param == "input":
            if typ == "range":
                return [1,100]
            if typ == "min":
                return 1
        elif param == "hidden":
            if typ == "range":
                return [1,1]
            if typ == "min":
                return 1
        elif param == "output":
            if typ == "range":
                return [1,100]
            if typ == "min":
                return 1
        elif param == "activation" or param == "dropout":
            if typ == "range":
                return [1,1]
            if typ == "min":
                return 1
        elif param == "activation_function":
            if typ == "range":
                return ['tanh']
            else:
                return None
        elif param == "optimizer":
            if typ == "range":
                return ['RMSprop']
            else:
                return None
        elif param == "batch_size":
            if typ == "range":
                return [64, 64]
            if typ == "min":
                return 8
        elif param == "epochs":
            if typ == "range":
                return [2, 2]
            if typ == "min":
                return 2
        elif param == "dropout_rate":
            if typ == "range":
                return [0, 0]
            if typ == "min":
                return 0.1
        elif param == "output_activation_function":
            if typ == "range":
                return ['sigmoid']
            else:
                return None
        else:
            raise ValueError("Unrecongised parameter: %s" % param)
    
    
    
    ## Testing end
        
    
    def initialize_mode_settings(self, y_train, y_test, y_val, loss, metric, test_metric, test_metric_direction):
        """
        From the label data,
        determines the number of output cells
        required in the model and store it
        """
        try:
            number_y_columns = y_train.shape[1]
        except AttributeError:
            try:
                number_y_columns = np.array(y_train).shape[1]
            except:
                raise TypeError("Unable to determine label data shape")
        except IndexError:
            number_y_columns = 1
        if self.data["mode"] == "classifier":
            if test_metric_direction != None:
                if test_metric_direction != "positive" and test_metric_direction != "negative":
                    raise ValueError("Unrecognised value for 'test_metric_direction'. Allowed values are 'positive' and 'negative'")
                self.data["test_metric_direction"] = test_metric_direction
            else:
                self.data["test_metric_direction"] = "positive"
            if number_y_columns == 1:
                unique = np.unique(y_train)
                if len(unique) < 2:
                    raise ValueError("minimum two unique labels required in training data")
                elif len(unique) == 2:
                    self.data["classifier_type"] = "single"
                    if 0 in unique and 1 in unique:
                        self.y_train = y_train
                        self.y_test = y_test
                        if y_val is not None:
                            self.y_val = y_val
                        self.data["label_encoded"] = False
                    else:
                        label_encoder = LabelEncoder().fit(y_train)
                        self.y_train = label_encoder.transform(y_train)
                        self.y_test = label_encoder.transform(y_test)
                        if y_val is not None:
                            self.y_val = label_encoder.transform(y_val)
                        self.data["label_encoded"] = True
                        self.data["label_encoder"] = label_encoder
                        
                    self.data["output_layer_units"] = 1
                    if loss != None:
                        self.loss = loss
                    else:
                        self.loss = "binary_crossentropy"
                    if metric != None:
                        self.metric = metric
                    else:
                        self.metric = "binary_accuracy"
                    if test_metric != None:
                        self.test_metric = test_metric
                    else:
                        self.test_metric = accuracy_score
                    ### Test for non 1-0 labels
                else:
                    self.data["classifier_type"] = "multi"
                    try:
                        self.y_train = np_utils.to_categorical(y_train)
                        self.y_test = np_utils.to_categorical(y_test)
                        if y_val is not None:
                            self.y_val = np_utils.to_categorical(y_val)
                        self.data["label_encoded"] = False
                        self.data["output_layer_units"] = self.y_train.shape[1]
                    except ValueError:
                        label_encoder = LabelEncoder().fit(y_train)
                        self.y_train = np_utils.to_categorical(label_encoder.transform(y_train))
                        self.y_test = np_utils.to_categorical(label_encoder.transform(y_test))
                        if y_val is not None:
                            self.y_val = np_utils.to_categorical(label_encoder.transform(y_val))
                        self.data["label_encoded"] = True
                        self.data["label_encoder"] = label_encoder
                        self.data["output_layer_units"] = self.y_train.shape[1]
                    if loss != None:
                        self.loss = loss
                    else:
                        self.loss = "categorical_crossentropy"
                    if metric != None:
                        self.metric = metric
                    else:
                        self.metric = "categorical_accuracy"
                    if test_metric != None:
                        self.test_metric = test_metric
                    else:
                        self.test_metric = accuracy_score
            elif number_y_columns > 1:
                self.data["classifier_type"] = "multi"
                self.y_train = y_train
                self.y_test = y_test
                if y_val is not None:
                    self.y_val = y_val
                self.data["label_encoded"] = False
                self.data["output_layer_units"] = number_y_columns
                if loss != None:
                    self.loss = loss
                else:
                    self.loss = "categorical_crossentropy"
                if metric != None:
                    self.metric = metric
                else:
                    self.metric = "categorical_accuracy"
                if test_metric != None:
                    self.test_metric = test_metric
                else:
                    self.test_metric = accuracy_score
            print(self.data["classifier_type"])
                
                        
        elif self.data["mode"] == "regressor":
            if test_metric_direction != None:
                if test_metric_direction != "positive" and test_metric_direction != "negative":
                    raise ValueError("Unrecognised value for 'test_metric_direction'. Allowed values are 'positive' and 'negative'")
                self.data["test_metric_direction"] = test_metric_direction
            else:
                self.data["test_metric_direction"] = "negative"
            if number_y_columns == 1:
                self.data["output_layer_units"] = 1
                self.y_train = y_train
                self.y_test = y_test
                if y_val is not None:
                    self.y_val = y_val
                if loss != None:
                    self.loss = loss
                else:
                    self.loss = "mean_absolute_error"
                if metric != None:
                    self.metric = metric
                else:
                    self.metric = "mae"
                if test_metric != None:
                    self.test_metric = test_metric
                else:
                    self.test_metric = mean_absolute_error
            else:
                raise ValueError("Unacceptable number of output values, %s, for regressor mode. Only one value allowed" % number_y_columns)
                
#         if type(self.y_test) is np.ndarray:
#             if self.y_test.dtype == 'int32':
#                 self.y_test = self.y_test.astype('float32')
        self.data["stage"] = 2
        ## Save data to local file
        print(self.metric)
        print(self.test_metric)
        print(self.loss)
        print(self.data["output_layer_units"])
        tools.save_data(self.data, self.data["directory"], self.data_filename)
        return None
        
    
    def get_param_type(self, param):
        """
        Returns the datatype for a given parameter
        """
        if "layer" in param:
            return "int"
        if param == "batch_size" or param == "epochs":
            return "int"
        if param == "dropout_rate":
            return "float"
        if param == "activation_function" or param == "output_activation_function" or param == "optimizer":
            return "str"
        raise ValueError("unrecognized paramaeter: %s" % param)

    def get_optimizer(self, name='Adadelta'):
        """
        Returns the optimizer given its name
        """        
        if name == 'Adam':
            return optimizers.Adam(clipnorm=1.)
        if name == 'Adadelta':
            return optimizers.Adadelta(clipnorm=1.)
        if name == 'SGD':
            return optimizers.SGD(clipnorm=1.)
        if name == 'Nadam':
            return optimizers.Nadam(clipnorm=1.)
        if name == 'RMSprop':
            return optimizers.RMSprop(clipnorm=1.)
        if name == 'Adagrad':
            return optimizers.Adagrad(clipnorm=1.)
        if name == 'Adamax':
            return optimizers.Adamax(clipnorm=1.)

        return optimizers.Adam(clipnorm=1.)
        
    def train(self, x_train, y_train, x_val=None, y_val=None, params=None, loss=None, metric=None):
        """Training function.
        Initializes a sequential network, 
        Fits the data using given parameters
        # Arguments
            x_train: training data
            y_train: training label
            x_val: validation data
            y_val: validation label
            params: the combination of parameters chosen for the iteration
            loss: loss function of the network
            metric: metric used to measure the performance of the network
        """
    
        ## Initialize Sequential network
        model = Sequential()
        kernel_initializer='normal'
        if self.verbose >= 2:
            verbose = self.verbose - 1
        else:
            verbose = 0
        
        ## Add layers to the model
        for i in range(0, len(self.data["layers"])):
            layer = layers.Layer(layer_type=self.data["layers"][i])
            if i == 0:
                count = 0
                model = layer.add_to_model(model, params, count+1, input_dim=x_train.shape[1])
            else:
                count = self.data["layers"][:i].count(self.data["layers"][i])
                if i == len(self.data["layers"]) - 1:
                    model = layer.add_to_model(model, params, count+1, output_layer_units=self.data["output_layer_units"], mode=self.data["mode"])
                else:
                    model = layer.add_to_model(model, params, count+1)
        
        ## Compile the model
        if self.data["mode"] == "classifier":
            model.compile(loss=loss,
                          optimizer=self.get_optimizer(params["optimizer"]), 
                          metrics=[metric])
        elif self.data["mode"] == "regressor":
            model.compile(loss=loss,
                          optimizer=self.get_optimizer(params["optimizer"]))
        
        ## Fit the model with or without validation data
        if x_val is not None and y_val is not None:
            model.fit(x_train, 
                      y_train,
                      batch_size=params["batch_size"],
                      epochs=params["epochs"],
                      verbose=verbose,
                      validation_data=(x_val, y_val),
                      shuffle=True
                     )
        else:
            model.fit(x_train, 
                      y_train,
                      batch_size=params["batch_size"],
                      epochs=params["epochs"],
                      verbose=verbose,
                      shuffle=True
                     )
        
        del kernel_initializer
        if self.data["stage"] == 2:
            model.save(os.path.join(self.data["directory"], 
                                    str(self.data["name"] + "-base_model.h5")))
            self.data["stage"] = 3
        return model
        
    
    def list_from_range(self, p, rn):
        """
        Given a range for the values of the parameters,
        Returns the list of options for combinations
        """
        typ = self.get_param_type(p)
#         rn = self.data["base_range"][p]
        if typ == "int" or typ == "float":
            if len(rn["range"]) == 0 or len(rn["range"]) > 2:
                raise ValueError("wrong range length")
        if typ == "int":
            if len(rn["range"]) == 1:
                return rn["range"]
            mn = rn["min"]
            top = round(rn["range"][1])
            bottom = round(rn["range"][0])
            dif = top - bottom
            if dif < 0:
                raise ValueError("Second number in range should be greater than or equal to the first one")
            elif dif == 0:
                return [top]
            
            ## If interval of range is less than minimum required, return the range limits
            elif dif <= mn:
                if bottom == top:
                    return [top]
                else:
                    return [bottom, top]
            elif dif <= 2:
                return list(range(bottom,top+1))
            
            ## Return list of values equidistant from nearby ones
            else:
#                 delta = int(dif/2)
#                 return [bottom, round(((bottom + delta) + (top - delta))/2), top]
                
                ## Level based
                bins = self.level + 1
                delta = int(dif/bins)
                result = [bottom]
                value = bottom
                for i in range(1, bins):
                    if value != round(bottom + (delta * i)):
                        value = round(bottom + (delta * i))
                        result.append(value)
                result.append(top)
                return result
                
            
        elif typ == "float":
            if len(rn["range"]) == 1:
                return rn["range"]
            mn = rn["min"]
            top = rn["range"][1] + 0.0
            bottom = rn["range"][0] + 0.0
            dif = top - bottom
            if dif < 0:
                raise ValueError("Second number in range should be greater than or equal to the first one")
            elif dif == 0:
                return [top]
            
            ## If interval of range is less than minimum required, return the range limits
            elif dif <= mn:
                if bottom == top:
                    return [top]
                else:
                    return [bottom, top]
            
            ## Return list of values equidistant from nearby ones
            else:
#                 delta = dif/2
#                 return [bottom, bottom + delta, top]
            
                ## Level based
                bins = self.level + 1
                delta = dif/bins
                result = [bottom]
                value = bottom
                for i in range(1, bins):
                    if value != bottom + (delta * i):
                        value = bottom + (delta * i)
                        result.append(value)
                result.append(top)
                return result
        elif typ == "str":
            return rn["range"]
    
    def generate_combinations(self, d):
        """
        Given a grid that includes all the parameter values, 
        Outputs all parameter combinations possible
        """
        r = {}
        for key in d.keys():
            r[key] = list(map(str, d[key]))
        
        keys, values = zip(*r.items())
        combs = [dict(zip(keys, v)) for v in it.product(*values)]
        return self.process_combinations(combs)
    
    def process_combinations(self, combs):
        """
        Convert combinations' parameter values into the correct datatypes
        """
        r = []
        for comb in combs:
            c = {}
            for p in comb.keys():
                if p.startswith("input_layer") or p.startswith("hidden_layer") or self.get_param_type(p) == "int":
                    c[p] = int(comb[p])
                elif self.get_param_type(p) == "str":
                    c[p] = comb[p]
                elif self.get_param_type(p) == "float":
                    c[p] = float(comb[p])
            r.append(c)
        return r
    
    def get_unique_combinations(self, combs):
        """
        Remove non-unique combinations
        """
        new = []
        for comb in combs:
            if comb not in new:
                new.append(comb)
        return new
                
    def range_from_last(self, grid, b_params):
        """
        Given the best combination and its parent grid, 
        Calculates the narrowed range of parameter values,
        for the next round
        """
        new = {}
        for p in grid.keys():
            typ = self.get_param_type(p)
            if typ == "str":
                new[p] = {"range":[b_params[p]], "type":typ}
            else:
                minim = self.data["base_range"][p]["min"]
                loc = grid[p].index(b_params[p])
                new[p] = {"min":minim}
                if len(grid[p]) == 1:
                    new[p]["range"] = [grid[p][loc], grid[p][loc]]
                elif len(grid[p]) == 2:
                    new[p]["range"] = [b_params[p], b_params[p]]
                elif loc == 0:
                    bottom = grid[p][loc] - (grid[p][loc+1] - grid[p][loc])/2
                    if bottom <= self.data["base_range"][p]["range"][0]:
                        new[p]["range"] = [grid[p][loc], (grid[p][loc] + grid[p][loc+1])/2]
                    else:
                        new[p]["range"] = [bottom, (grid[p][loc] + grid[p][loc+1])/2]
                elif loc == len(grid[p]) - 1:
                    top = grid[p][loc] + (grid[p][loc] - grid[p][loc-1])/2
                    if top >= self.data["base_range"][p]["range"][1]:
                        new[p]["range"] = [(grid[p][loc] + grid[p][loc-1])/2, grid[p][loc]]
                    else:
                        new[p]["range"] = [(grid[p][loc] + grid[p][loc-1])/2, top]
                else:
                    new[p]["range"] = [(grid[p][loc] + grid[p][loc-1])/2, (grid[p][loc] + grid[p][loc+1])/2]
        return new
    
    def grid_from_comb(self, last, comb):
        """
        Given the combination,
        Returns the grid it was derived from
        """
        grids = self.data["grids"][last]
        for key in grids.keys():
            check = 0
            for key2 in grids[key].keys():
                if comb[key2] in grids[key][key2]:
                    check += 1
            if check == len(grids[key]):
                return grids[key]
        raise ValueError("grid not found")
                    
    
    def run_comb(self, comb):
        """
        Builds a model with the given combination of parameters,
        trains it and scores it on the test data.
        Returns the model and the score
        """
        for k in range(0, self.level+1):
            model = self.train(self.x_train, self.y_train, self.x_val, self.y_val, params=comb, loss=self.loss, metric=self.metric)
            yp_test = model.predict(self.x_test)
            if self.test_metric == accuracy_score:
                yp_test = yp_test.round()
            if k == 0:
                score = self.test_metric(self.y_test, yp_test)
            else:
                score = ((score * k) + self.test_metric(self.y_test, yp_test))/(k+1)
        del yp_test, k
        return model, score
        
    
    def run_round(self, n):
        """
        Builds the grids of parameter values for the round,
        finds all possible combinations from the grids,
        tries all combinations,
        stores the best combination(s)
        """
        n = str(n)
        
        ## If combinations have not been found already, find them
        try:
            self.data["combs"][n]
        except KeyError:
            
            ## if initial round, find combinations from base grid
            if int(n) == 1:
                self.data["grids"][n] = {}
                for p in self.data['base_range'].keys():
                    self.data["grids"][n][p] = self.list_from_range(p, self.data['base_range'][p])
                combinations = self.generate_combinations(self.data["grids"][n])
            
            ## Find combinations based on the best combinations from the last round
            elif int(n) >= 2:
                self.data["grids"][n] = {}
                last = str(int(n)-1)
                combinations = []
                for key in self.data["best"][last].keys():
                    if int(n) == 2:
                        grid = self.data["grids"][last]
                    else:
                        grid = self.grid_from_comb(last, self.data["best"][last][key])
                    rn = self.range_from_last(grid, self.data["best"][last][key])
                    self.data["grids"][n][key] = {}
                    for p in rn.keys():
                        self.data["grids"][n][key][p] = self.list_from_range(p, rn[p])
                    del rn
                for key in self.data["grids"][n].keys():
                    c = self.generate_combinations(self.data["grids"][n][key])
                    combinations = combinations + c
                combinations = self.get_unique_combinations(combinations)
            
            ## Initialize supporting variables and save data
            self.data["combs"][n] = combinations
            self.data["combs_comp"][n] = [False] * len(combinations)
            self.data["best"][n] = {}
            tools.save_data(self.data, self.data["directory"], self.data_filename)
            del combinations
        
        ## Check if all combinations have been tried
        if len(self.data["combs"][n]) == sum(self.data["combs_comp"][n]):
            return self
#         print("Round grid:")
#         print(self.data["grids"][n])
        print("%s new combinations found. Trying them." % (len(self.data["combs"][n]) - sum(self.data["combs_comp"][n])))
        
        ## Try each combination
        for i in range(0, len(self.data["combs"][n])):
            comb = self.data["combs"][n][i]

            if self.data["combs_comp"][n][i] == True:
                continue
            
            if self.verbose >= 1:
                print("Trying combination: %s" % str(i+1))
#                 print(datetime.datetime.now())
            if self.verbose >= 2:
                print(comb)
            model, score = self.run_comb(comb)
            score = float(round(score,8))
            
            ## If combination's score is amongst the best, store it
            ## If no best scores for the round are present, add score to best scores
            if len(self.data["best"][n]) < self.level + 1:
                if str(score) not in self.data["best"][n].keys():
                    print("Best1")
                    print(self.data["best"][n])
                    print(self.level)
                    print(score)
                    self.data["best"][n][str(score)] = comb
            ## If best scores for the round are present, compare with the least score amongst them and store the current score if it is better than the least
            else:
                if self.data["test_metric_direction"] == "positive":
                    min_score = sorted(map(float, self.data["best"][n].keys()))[0]
                    if score > min_score:
                        print("Best2")
                        self.data["best"][n][str(score)] = comb
                        del self.data["best"][n][str(min_score)]
                elif self.data["test_metric_direction"] == "negative":
                    max_score = sorted(map(float, self.data["best"][n].keys()))[len(self.data["best"][n].keys()) - 1]
                    if score < max_score:
                        print("Best2")
                        self.data["best"][n][str(score)] = comb
                        del self.data["best"][n][str(max_score)]
            
            ## If no overall best scores are present, add score to best scores
            if len(self.data["best"]["best"]) < self.level + 1:
                if str(score) not in self.data["best"]["best"].keys():
                    print("Best3")
                    print(self.data["best"]["best"])
                    print(self.level)
                    print(score)
                    self.data["best"]["best"][str(score)] = comb
                    model.save(os.path.join(self.data["directory"], 
                                            str(self.data["name"] + "-model-" + str(score) + "-s.h5")))
                    if self.data["stage"] == 3:
                        self.data["stage"] = 4
            ## If overall best scores are present, compare with the least score amongst them and store the current score, model if it is better than the least
            else:
                if self.data["test_metric_direction"] == "positive":
                    min_score = sorted(map(float, self.data["best"]["best"].keys()))[0]
                    if score > min_score:
                        print("Best4")
                        self.data["best"]["best"][str(score)] = comb
                        model.save(os.path.join(self.data["directory"], 
                                                str(self.data["name"] + "-model-" + str(score) + "-s.h5")))
                        del self.data["best"]["best"][str(min_score)]
                        try:
                            os.remove(os.path.join(self.data["directory"], 
                                                   str(self.data["name"] + "-model-" + str(min_score) + "-s.h5")))
                        except FileNotFoundError:
                            pass
                elif self.data["test_metric_direction"] == "negative":
                    max_score = sorted(map(float, self.data["best"]["best"].keys()))[len(self.data["best"][n].keys()) - 1]
                    if score < max_score:
                        print("Best4")
                        self.data["best"]["best"][str(score)] = comb
                        model.save(os.path.join(self.data["directory"], 
                                                str(self.data["name"] + "-model-" + str(score) + "-s.h5")))
                        del self.data["best"]["best"][str(max_score)]
                        try:
                            os.remove(os.path.join(self.data["directory"], 
                                                   str(self.data["name"] + "-model-" + str(max_score) + "-s.h5")))
                        except FileNotFoundError:
                            pass

            self.data["combs_comp"][n][i] = True
            tools.save_data(self.data, self.data["directory"], self.data_filename)
            
            ## Clear memory
            del comb, model, score
            gc.collect()
            K.clear_session()
            
        
        if self.data["stage"] == 4:
            self.data["stage"] = 5
        if len(self.data["combs"][n]) == 1:
            self.data["optimized"] = True
            self.data["stage"] = 9
        
        return self
    
    def optimize(self, x_train, y_train, x_test, y_test, x_val=None, y_val=None, loss=None, metric=None, test_metric=None, test_metric_direction=None, verbose=1, max_rounds=1, level=1):
        """Main optimization function.
        Checks data to find the status of optimization, 
        Runs each round of optimization
        # Arguments
            x_train: training data
            y_train: training label
            x_test: testing data
            y_test: testing label
            x_val: validation data
            y_val: validation label
            params: the combination of parameters chosen for the iteration
            loss: loss function of the network
            metric: metric used to measure the performance of the network
            test_metric: metric used to measure the model's performance on the test data
            test_metric_direction: direction of the test metric. Allowed values:
                                    "positive", if higher values are better(example: accuracy) or 
                                    "negative", if lower values are better(example:error)
            verbose: level of verbosity of the output shown to the user
            max_rounds: maximum rounds of optimization to be run
            level: level of intensity optimization
        """
        print("Initializing optimizer settings")
        self.initialize_mode_settings(y_train, y_test, y_val, loss, metric, test_metric, test_metric_direction)
        self.x_train = x_train
#         self.y_train = y_train
        self.x_test = x_test
#         self.y_test = y_test
        self.x_val = x_val
#         self.y_val = y_val
#         self.loss = loss
#         self.metric = metric
#         self.test_metric = test_metric
        self.verbose = verbose
        self.level = level
        self.data_filename = str(self.data["name"] + "-data.pickle")
                
        
        ## Check how many rounds and combinations have been run
        rounds = list(map(int, self.data["combs"].keys()))
        if len(rounds) == 0:
            c_round = 1
        else:
            rounds.sort()
            c_round = rounds[len(rounds)-1]
        
        ## Run each round of optimization
        while c_round <= max_rounds:
            if self.data["optimized"] == True:
                print("Best parameters found")
                break
            print("Running round: %s" % c_round)
            self.run_round(c_round)
            gc.collect()
            c_round += 1
        print("%s rounds of optimization completed" % str(c_round-1))
        return self
        
    def get_data(self):
        """
        Returns the data stored by the optimizer
        """
        return self.data


# In[6]:


def predict(x_predict, optimizer_name="donn_optimizer", directory=os.getcwd(), probabilities=False, use_one_model=False):
    """
    Given the data(x values) for prediction,
    loads the best models, predicts using them,
    returns the average of all predictions
    weighted by the model scores
    Arguments:
        x_predict: data for which predictions are to be made.
            Must have the same dimensions as training data used by the optimizer(x_train)
        optimizer_name: name of the optimizer to use for prediction
        directory: file directory where the optimizer data is stored
    """
    ## Read data
    data_filename = str(optimizer_name + "-data.pickle")
    data = tools.read_data(directory, data_filename)
    
    ## If no training has been done output information
    if data["stage"] <= 2:
        print("Optimizer cannot predict before training")
        return None
    
    ## if the firt model has been trained and no best model has been found use the first model
    elif data["stage"] == 3:
        model = keras_load_model(os.path.join(data["directory"], 
                                              str(data["name"] + "-base_model.h5")))
        prediction = model.predict(x_predict)
        return model.predict(x_predict)

    ## if best models have been found, use them
    else:
        models = {}
        for key in data["best"]["best"].keys():
            models[key] = {}
            try:
                models[key]["model"] = keras_load_model(os.path.join(data["directory"], 
                                                                     str(data["name"] + "-model-" + key + "-s.h5")))
            except OSError:
                models.pop(key, None)
                continue
            except FileNotFoundError:
                models.pop(key, None)
                continue

        if len(models) == 0:
            print("no stored models found")
            return None
        
        if use_one_model == True:
            if data["test_metric_direction"] == "positive":
                key = str(sorted(map(float, data["best"]["best"].keys()))[0])
            else:
                key = str(sorted(map(float, data["best"]["best"].keys()))[len(data["best"]["best"]) - 1])
            prediction = models[key]["model"].predict(x_predict)
        ## Return average of the predictions from each model weighted by the model scores
        else:
            i = 0
            for key in models.keys():
                if float(key) == 0:
                    weight = 0
                elif data["test_metric_direction"] == "negative":
                    weight = 1/float(key)
                elif data["test_metric_direction"] == "positive":
                    weight = float(key)
                models[key]["prediction"] = models[key]["model"].predict(x_predict) * weight
                if i == 0:
                    total = models[key]["prediction"]
                    denom = weight
                    i += 1
                else:
                    total = total + models[key]["prediction"]
                    denom = denom + weight
            if denom == 0:
                prediction = total
            else:
                prediction = total / denom
            
        if data["mode"] == "classifier":
            if probabilities == True:
                if data["label_encoded"] == True:
                    return data["label_encoder"].classes_ , prediction
            if data["classifier_type"] == "single":
                prediction = np.round(prediction).astype('int32')
            elif data["classifier_type"] == "multi":
                prediction = prediction.argmax(axis=1)            
            if data["label_encoded"] == True:
                prediction = data["label_encoder"].inverse_transform(prediction)
        return prediction


# In[7]:


from sklearn.datasets import load_boston


# In[8]:


X, Y = load_boston(return_X_y=True)


# In[11]:


cut_v = round(X.shape[0] * 0.8)
cut_t = round(X.shape[0] * 0.9)
cut_x = round(X.shape[0])
# cut_y = round(X.shape[0] * 0.11)
x_train = X[:cut_v]
y_train = Y[:cut_v]
x_val = X[cut_v:cut_t]
y_val = Y[cut_v:cut_t]
x_test = X[cut_t:]
y_test = Y[cut_t:]


# In[10]:


op = Optimizer(name="donn_optimizer", mode="regressor")


# In[12]:


op.optimize(x_train,
            y_train,
            x_test,
            y_test,
            x_val,
            y_val,
            verbose=1,
            max_rounds=3,
            level=2
           )


# In[13]:


op.data


# In[11]:


with open('X.pickle', 'rb') as f:
    X = pickle.load(f)
with open('Y_multi.pickle', 'rb') as f:
    Y = pickle.load(f)


# In[12]:


cut_v = round(X.shape[0] * 0.8)
cut_t = round(X.shape[0] * 0.9)
cut_x = round(X.shape[0])
# cut_y = round(X.shape[0] * 0.11)
x_train = X[:cut_v]
y_train = Y[:cut_v]
x_val = X[cut_v:cut_t]
y_val = Y[cut_v:cut_t]
x_test = X[cut_t:]
y_test = Y[cut_t:]
# x_test2 = X[cut_x:cut_y]
# y_test2 = Y[cut_x:cut_y]


# In[22]:


y_val.shape


# In[49]:


with open('data.pickle', 'wb') as f:
    pickle.dump(data, f)


# In[43]:


op = Optimizer(name="donn_optimizer", mode="regressor")


# In[48]:


op2 = op.optimize(x_train, 
                   y_train, 
                   x_test, 
                   y_test, 
                   x_val, 
                   y_val,
                   verbose=1,
                   max_rounds=3,
                   level=2
                  )


# In[47]:


op2.data


# In[49]:


op2.data


# In[50]:


p = predict(x_test)


# In[51]:


p


# In[52]:


mean_absolute_error(p, y_test)


# In[268]:


accuracy_score(p.round(), np_utils.to_categorical(y_test))


# In[229]:


from pymongo import MongoClient
import scipy.sparse as sp


# In[232]:


import utils_m


# In[233]:


client = MongoClient()
db_stockml = client["db_stockml"]
c_train_data_yahoofin = db_stockml["c_train_data_yahoofin"]

l = "l_0_p5ds_abs"
print(datetime.datetime.now())
print("Using Label: %s" % l)
raw = list(db_stockml.c_train_data_yahoofin.find({l:{"$exists":True}}))
print("Prepping %s datapoints" % len(raw))
a = raw[0]
csr_a_t2 = sp.csr_matrix((a["t2_data"], a["t2_indices"], a["t2_indptr"]))
csr_a_t1 = sp.csr_matrix((a["t1_data"], a["t1_indices"], a["t1_indptr"]))
X = utils_m.add_sparse(csr_a_t2, csr_a_t1)
y = [a[l]]
for i in range(1,round(len(raw)/10)):

    a = raw[i]
    if len(a["t2_data"]) == 0 or len(a["t1_data"]) == 0:
        continue
    csr_t2 = sp.csr_matrix((a["t2_data"], a["t2_indices"], a["t2_indptr"]))
    csr_t1 = sp.csr_matrix((a["t1_data"], a["t1_indices"], a["t1_indptr"]))
    csr = utils_m.add_sparse(csr_t2, csr_t1)
    x = utils_m.vstack_dim(X, csr)
    if x == None:
        continue
    else:
        X = x.tocsr()
    y.append(a[l])

client.close()


# In[238]:


def logist(x):
    return 1/(1+np.exp(-x))


# In[361]:


Y = y_cont


# In[345]:


yc = []
y_cont = []
cut_h = np.mean(y) + (np.std(y)/8)
cut_l = np.mean(y) - (np.std(y)/8)
for i in range(0,len(y)):
    y_cont.append(logist(abs(y[i]))+0.25)
#     if y[i] > cut_h:
#         yc.append("c")
#     elif y[i] > cut_l:
#         yc.append("b")
#     else:
#         yc.append("a")
    if y[i] > 0:
        yc.append("a")
    else:
        yc.append("b")
Y = np.transpose([yc])
y_cont = abs(np.transpose([y_cont]))


# In[288]:


with open('X.pickle', 'wb') as f:
    pickle.dump(X,f)
with open('Y_multi_labeled.pickle', 'wb') as f:
    pickle.dump(Y,f)
with open('Y_cont.pickle', 'wb') as f:
    pickle.dump(y_cont,f)


# In[324]:


a = ["a", "b", "a", "b"]
le = LabelEncoder().fit(a)
le.transform(a)


# In[325]:


le.classes_

