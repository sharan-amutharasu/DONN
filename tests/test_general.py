
# coding: utf-8

# In[ ]:


try:
    import pytest
    import os
    import sys
    from numpy import unique

    from sklearn.metrics import accuracy_score, mean_absolute_error

    import donn
    from donn import donn_tools

    from sklearn.datasets import load_digits # Multi label classification
    from sklearn.datasets import load_breast_cancer # Single label classificaiton
    from sklearn.datasets import load_diabetes # Regression
except ImportError:
    print("Required modules not installed")


# In[2]:


datasets = {"load_digits":{"name":"digits", 
                           "fetch_data_function":load_digits, 
                           "mode":"classifier", 
                           "classifier_type":"multi"},
            "load_breast_cancer":{"name":"breast_cancer", 
                                  "fetch_data_function":load_breast_cancer, 
                                  "mode":"classifier", 
                                  "classifier_type":"single"},
            "load_diabetes":{"name":"diabetes", 
                             "fetch_data_function":load_diabetes, 
                             "mode":"regressor", 
                             "classifier_type":None}}


# In[3]:


def test_mode(validation=True):
    """
    Tests optimizer for different types of prediction tasks such as regression, single and multi label classification.
    Test if predictions are made and if optimized model results are better than unoptimized model
    """
    for dataset in datasets.keys():
        print(datasets[dataset]['name'])
        X, Y = datasets[dataset]['fetch_data_function'](return_X_y=True) # Get data
        x_train, y_train, x_validation, y_validation, x_test, y_test, x_supertest, y_supertest = donn_tools.split_dataset(X, Y, validation=True, supertest=True) # Split data
        
        
        ## Run Optimizer
        optimizer = donn.Optimizer(name = datasets[dataset]['name'], mode = datasets[dataset]['mode'])
        if validation == True:
            optimizer.optimize(x_train, y_train, x_validation, y_validation, x_test, y_test, max_rounds=1, level=1)
        else:
            optimizer.optimize(x_train, y_train, x_test, y_test, max_rounds=1, level=1)
        
        y_prediction = donn.predict(x_supertest, optimizer_name = datasets[dataset]['name']) # make prediction
        
        assert x_supertest.shape[0] == y_prediction.shape[0] # Check if predictions are present for each datapoint
        
        assert len(unique(y_prediction)) > 0 # Check if predictions are different
        
        ## Get baseline score for comparison
        if validation == True:
            baseline_score = donn.run_base_model(x_train, y_train, x_val=x_validation, y_val=y_validation, x_test=x_supertest, y_test=y_supertest, mode=datasets[dataset]['mode'], classifier_type=datasets[dataset]['classifier_type'])
        else:
            baseline_score = donn.run_base_model(x_train, y_train, x_test=x_supertest, y_test=y_supertest, mode=datasets[dataset]['mode'], classifier_type=datasets[dataset]['classifier_type'])
        
        ## Check if optimized results are better than baseline results
        if datasets[dataset]['mode'] == 'classifier':
            score = accuracy_score(y_supertest, y_prediction)
            assert score >= baseline_score
        elif datasets[dataset]['mode'] == 'regressor':
            score = mean_absolute_error(y_supertest, y_prediction)
            assert score <= baseline_score
    return "Test completed"


# In[ ]:


if __name__ == '__main__':
    test_mode(validation=True)
    test_mode(validation=False)

