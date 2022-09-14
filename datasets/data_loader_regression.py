
from tensorflow import keras
import numpy as np


def data_loader_fc_regression():
    (x, y), (x_test, y_test) = keras.datasets.boston_housing.load_data()
        
    x_train = x[:-300]
    x_val = x[-300:]
    y_train = y[:-300]
    y_val = y[-300:]
        
    return x_train,x_test,x_val,y_train,y_test,y_val
