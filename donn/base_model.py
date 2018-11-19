
# coding: utf-8

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.utils import to_categorical

from sklearn.metrics import accuracy_score, mean_absolute_error


# In[2]:


def run_base_model(x_train, y_train, x_val=None, y_val=None, mode=None, x_test=None, y_test=None, classifier_type=None):
    """
    Creates a basic unoptimized network and returns its score,
    to be used as benchmark for comparison of optimized models
    """
    
    ## Build model
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=x_train.shape[1], kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(100, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(100, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.01)))
    try:
        if mode == "classifier" and classifier_type == "multi":
            output_cells = to_categorical(y_train).shape[1]
        else:
            output_cells = y_train.shape[1]
    except IndexError:
        output_cells = 1
    if mode == "classifier":
        model.add(Dense(output_cells, kernel_initializer='normal', activation='sigmoid'))
    else:
        model.add(Dense(output_cells, kernel_initializer='normal'))
    
    ## Compile model
    if mode == "classifier":
        if classifier_type == "single":
            model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        elif classifier_type == "multi":
            y_train = to_categorical(y_train)
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        else:
            raise ValueError("Unrecognised 'classifier_type'")
    elif mode == "regressor":
        model.compile(loss='mae', optimizer='rmsprop')
    else:
        raise ValueError("unrecognised 'mode'")
    
    ## Train model
    if x_val is not None and y_val is not None:
        if mode == "classifier" and classifier_type == "multi":
            y_val = to_categorical(y_val)
        model.fit(x_train, y_train, batch_size=128, epochs=5, verbose=0, validation_data=(x_val, y_val), shuffle=True)
    else:
        model.fit(x_train, y_train, batch_size=128, epochs=5, verbose=0, shuffle=True)
    
    ## Make prediction
    y_prediction = model.predict(x_test)
    
    ## Score prediction
    if mode == "classifier":
        y_prediction = y_prediction.round()
        try:
            score = accuracy_score(y_test, y_prediction)
        except ValueError:
            score = accuracy_score(to_categorical(y_test), y_prediction)
    else:
        score = mean_absolute_error(y_test, y_prediction)
    
    return score

