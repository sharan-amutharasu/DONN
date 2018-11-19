
# coding: utf-8

# In[20]:


import pickle
import time
import string
import random
import os


# In[21]:


def create_flag_file(flag_file):
    """
    Creates a flag file to indicate read/write status of data file
    """
    with open(flag_file, "w") as f:
        f.write("close")
    return None


# In[22]:


def save_data(data, folder, filename):
    """
    Saves data to file after checking if the file is already 
    being accessed by another program using the flag file
    """
    flag_file = os.path.join(folder, str("flag_" + filename + ".txt"))
    data_file = os.path.join(folder, filename)
    try:
        with open(flag_file, "r") as g:
            check = g.read()
    except FileNotFoundError:
        create_flag_file(flag_file)
        with open(flag_file, "r") as g:
            check = g.read()
    if check == "open":
        print("file %s open" % filename)
        time.sleep(5)
        return save_data(data, folder, filename)
    with open(flag_file, "w") as f:
        f.write("open")
    with open(flag_file, "r") as g:
        check = g.read()
    if check == "close":
        print("file %s open" % filename)
        time.sleep(5)
        return save_data(data, folder, filename)
    with open(data_file, 'wb') as f:
        pickle.dump(data, f)
    with open(flag_file, "w") as f:
        f.write("close")
    return None


# In[23]:


def read_data(folder, filename):
    """
    Reads and returns the data from the data file after checking 
    if the file is already being accessed by another program
    using the flag file
    """
    flag_file = os.path.join(folder, str("flag_" + filename + ".txt"))
    data_file = os.path.join(folder, filename)
    with open(flag_file, "r") as g:
        check = g.read()
    if check == "open":
        print("file %s open" % filename)
        time.sleep(5)
        return read_data(folder, filename)
    with open(flag_file, "w") as f:
        f.write("open")
    with open(flag_file, "r") as g:
        check = g.read()
    if check == "close":
        print("file %s open" % filename)
        time.sleep(5)
        return read_data(folder, filename)
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    with open(flag_file, "w") as f:
        f.write("close")
    return data


# In[24]:


def random_string(n):
    """
    Generates a random string of given length out of 
    upercase, lowercase alphabets and, numerals
    """
    base = string.ascii_letters + string.digits
    r = ""
    for i in range(0,n):
        r = r + base[random.randrange(0,len(base))]
    return r

def generate_run_id(db_name):
    return random_string(16)


# In[25]:


def accuracy(y_true, y_pred):
    """
    Backup accuracy function
    """
    p_count = 0
    n_count = 0
    for i in range(0,len(y_pred)):
        if y_pred[i] >= 0.5:
            if y_true[i] >= 0.5:
                p_count += 1
            else:
                n_count += 1
        elif y_pred[i] < 0.5:
            if y_true[i] < 0.5:
                p_count += 1
            else:
                n_count += 1
    if p_count + n_count == 0:
        return 0
    else:
        return p_count/(p_count + n_count)


# In[2]:


def split_dataset(X, Y, validation = True, supertest = True, train_cut = 0.7, validation_cut = 0.8, test_cut = 0.9):
    """
    Splits dataset into train, test, validation and supertest categories
    """
    train_cut = round(X.shape[0] * train_cut)
    validaiton_cut = round(X.shape[0] * validation_cut)
    test_cut = round(X.shape[0] * test_cut)
    x_train = X[:train_cut]
    y_train = Y[:train_cut]
    if validation == True:
        x_validation = X[train_cut:validaiton_cut]
        y_validation = Y[train_cut:validaiton_cut]
        x_test = X[validaiton_cut:test_cut]
        y_test = Y[validaiton_cut:test_cut]
            
    else:
        x_validation = None
        y_validation = None
        x_test = X[train_cut:test_cut]
        y_test = Y[train_cut:test_cut]
        
    if supertest == True:
        x_test = X[train_cut:test_cut]
        y_test = Y[train_cut:test_cut]
        x_supertest = X[test_cut:]
        y_supertest = Y[test_cut:]
    else:
        x_test = X[train_cut:]
        y_test = Y[train_cut:]
        x_supertest = None
        y_supertest = None
        
    return x_train, y_train, x_validation, y_validation, x_test, y_test, x_supertest, y_supertest

