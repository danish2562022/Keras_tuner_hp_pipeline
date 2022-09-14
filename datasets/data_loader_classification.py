
from tensorflow import keras
import numpy as np



def load_data():
    
    (x, y), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x[:-10000]
    x_val = x[-10000:]
    y_train = y[:-10000]
    y_val = y[-10000:]

    x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
    x_val = np.expand_dims(x_val, -1).astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0

    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    
    return x_train,x_test,x_val,y_train,y_test,y_val

        
    







