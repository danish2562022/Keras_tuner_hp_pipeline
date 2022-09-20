from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers


    

p = {
        "min_number_of_layers": 1,
        "max_number_of_layers": 5,
        "model_type" : "r",    # "r" for regression and "c" for classification
        "min_units_per_layers" : 32,
        "max_units_per_layers" : 512,
        "num_of_classes" : 10,
        "choose_optimizer": "adam",
        "batch_size" : [32,64,128]
        
    }


def build_model(self,hp):
   
    model = keras.Sequential()
    
    if p['model_type'] == 'c':
        model.add(layers.InputLayer(input_shape=(28,28,1)))
        model.add(layers.Flatten())
    else:
        model.add(layers.InputLayer(input_shape=(13,1)))
    
    for i in range(1,hp.Int("num_layers",p['min_number_of_layers'],p['max_number_of_layers'])+1):
       
        model.add(
            layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=p['min_units_per_layers'], max_value=p['max_units_per_layers'], step=32),
                kernel_regularizer = tf.keras.regularizers.L2(l2=hp.Float(f"lr_{i}", min_value=1e-4, max_value=1e-2, sampling="log")),
                activation=hp.Choice(f"activation_{i}", ["relu", "tanh"]),
            )
        )
        if hp.Boolean(f"dropout_{i}"):
            model.add(layers.Dropout(rate=0.25))
    
    if p['model_type'] == 'r':
        model.add(layers.Dense(1))
        
    else:
        model.add(layers.Dense(p['num_of_classes'], activation="softmax"))
        
    return model

def load_data():

    if p["model_type"] == "c":
    
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

    elif p["model_type"] == "r":
        (x, y), (x_test, y_test) = keras.datasets.boston_housing.load_data()
            
        x_train = x[:-300]
        x_val = x[-300:]
        y_train = y[:-300]
        y_val = y[-300:]    
        return x_train,x_test,x_val,y_train,y_test,y_val

        
