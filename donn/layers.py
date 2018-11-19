
# coding: utf-8

# In[4]:


from keras.layers import Activation, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU, PReLU, ThresholdedReLU, ELU
from keras import regularizers


# In[5]:


def get_activation_layer(activation):
    """
    Returns the activation layer given its name
    """
    if activation == 'ELU':
        return ELU()
    if activation == 'LeakyReLU':
        return LeakyReLU()
    if activation == 'ThresholdedReLU':
        return ThresholdedReLU()
    if activation == 'PReLU':
        return PReLU()

    return Activation(activation)


# In[4]:


class Layer(object):
    """
    Layer object for adding different types of layers to the model
    """
    def __init__(self, layer_type):
        self.layer_type = layer_type
        if self.layer_type in ["hidden", "input", "output"]:
            self.kernel_initializer='normal'
            self.kernel_regularizer=regularizers.l2(0.01)
        
    def add_to_model(self, model, params, count, input_dim=None, output_layer_units=None, mode=None, layers=None):
        """
        Add layer to model
        """
        ## Input Layer
        if self.layer_type == "input":
            units = params[str(self.layer_type + "_layer_" + str(count) + "_units")]
            if input_dim is not None:
                model.add(Dense(units, input_dim=input_dim, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer))
            else:
                model.add(Dense(units, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer))
            return model
        
        ## Hidden Layer
        if self.layer_type == "hidden":
            units = params[str(self.layer_type + "_layer_" + str(count) + "_units")]
            if input_dim is not None:
                model.add(Dense(units, input_dim=input_dim, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer))
            else:
                model.add(Dense(units, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer))
            return model
        
        ## Activation Layer
        if self.layer_type == "activation":
            model.add(get_activation_layer(params["activation_function"]))
            return model
        
        ## Dropout Layer
        if self.layer_type == "dropout":
            dropout_rate = params["dropout_rate"]
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
            return model
        
        ## Output Layer
        if self.layer_type == "output":
            if mode == "classifier":
                model.add(Dense(output_layer_units, kernel_initializer=self.kernel_initializer))
                try:
                    if params["output_activation_function"] != None:
                        model.add(get_activation_layer(params["output_activation_function"]))
                except KeyError:
                    pass
            elif mode == "regressor":
                model.add(Dense(output_layer_units, kernel_initializer=self.kernel_initializer))
            else:
                raise ValueError("mode has to be 'regressor' or 'classifier'")
            return model
        
        ## LSTM Layer
#         if self.layer_type == "LSTM":
#             units = params[str(self.layer_type + "_layer_" + str(count) + "_units")]
#             count_LSTM = layers.count("LSTM")
#             if count < count_LSTM:
#                 return_sequences = True
#             else:
#                 return_sequences = False
#             if input_dim is not None:
#                 model.add(LSTM(units, input_dim=input_dim, recurrent_activation=params["LSTM_recurrent_activation_function"], return_sequences=return_sequences))
#             else:
#                 model.add(LSTM(units, recurrent_activation=params["LSTM_recurrent_activation_function"], return_sequences=return_sequences))
#             return model

