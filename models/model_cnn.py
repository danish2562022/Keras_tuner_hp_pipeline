import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

p = {
        "min_number_of_conv_layers": 1,
        "max_number_of_conv_layers": 2,
        "model_type" : "c",
        "min_number_of_filters" : 32,
        "max_numbe_of_filters" : 128,
        "kernel_size": [3,5,7],
        "stride":[1,1],
        "max_pooling_size": [1,2],
        "min_number_of_dense_layers":1,
        "max_number_of_dense_layers": 3,
        "min_unit_dense_layer":32,
        "max_unit_dense_layer":512,
        "num_of_classes" : 10,
        "choose_optimizer": "adam",
        "training_data_size": 1000,
        "validation_data_size": 500        
    }


def build_model(self,hp):

    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=(28,28,1)))
    for i in range(1,hp.Int("num_of_conv_layers",p['min_number_of_conv_layers'],p['max_number_of_conv_layers'])+1):
        model.add(
                layers.Conv2D(
                 filters = hp.Int(f"conv_layer_{i}_filers", min_value=p['min_number_of_filters'], max_value=p['max_numbe_of_filters'], step=16),
                 kernel_size = hp.Choice(f"conv_layer_{i}_kernel_size",values=p['kernel_size']),
                 strides = hp.Choice(f"conv_layer_{i}_kernel_stride",values=p['stride']),
                 kernel_regularizer = tf.keras.regularizers.L2(l2=hp.Float(f"conv_layer_{i}_lr", min_value=1e-4, max_value=1e-2, sampling="log")),
                 activation=hp.Choice(f"conv_layer_{i}_activation", ["relu", "tanh"])   
                )
        )
        if hp.Boolean(f"Batch_Norm_after_conv_layer_{i}"):
            model.add(layers.BatchNormalization())
        if hp.Boolean(f"dropout_after_conv_layer_{i}"):
            model.add(layers.Dropout(rate=0.25))
        
        model.add(layers.MaxPooling2D(pool_size = hp.Choice(f"maxpooling_layer_{i}_pool_size",values=p['max_pooling_size'])))

    model.add(layers.Flatten())

    for j in range(i+1,hp.Int("num_of_dense_layers_after_conv",p['min_number_of_dense_layers']+i,p['max_number_of_dense_layers']+i)+1):
       
        model.add(
            layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"dense_layer_{j}_units", min_value=p['min_unit_dense_layer'], max_value=p['max_unit_dense_layer'], step=32),
                kernel_regularizer = tf.keras.regularizers.L2(l2=hp.Float(f"dense_layer_{j}_lr", min_value=1e-4, max_value=1e-2, sampling="log")),
                activation=hp.Choice(f"dense_layer_{j}_activation", ["relu", "tanh"]),
            )
        )
        if hp.Boolean(f"dropout_after_dense_layer_{j}"):
            model.add(layers.Dropout(rate=0.25))
    
    model.add(layers.Dense(p['num_of_classes'], activation="softmax"))



    return model


def load_data():
    
    (x, y), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x[:p['training_data_size']]
    x_val = x[-p['validation_data_size']:]
    y_train = y[:p['training_data_size']]
    y_val = y[-p['validation_data_size']:]

    x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
    x_val = np.expand_dims(x_val, -1).astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0

    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    
    return x_train,x_test,x_val,y_train,y_test,y_val

        
    
