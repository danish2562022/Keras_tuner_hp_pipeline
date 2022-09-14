import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

p = {
        "min_number_of_conv_layers": 1,
        "max_number_of_conv_layers": 5,
        "model_type" : "c",
        "min_number_of_filters" : 32,
        "max_numbe_of_filters" : 128,
        "kernel_size": [3,5,7],
        "min_number_of_dense_layers":1,
        "max_number_of_dense_layers": 3,
        "num_of_classes" : 10,
        "choose_optimizer": "adam",
        
    }


def build_model(self,hp):

    model = keras.Sequential()

    for i in range(1,hp.Int("num_of_conv_layers",p['min_number_of_conv_layers'],p['max_number_of_conv_layers'])+1):
        model.add(
                layers.Conv2D(
                 filters = hp.Int(f"conv_1_filers_{i}", min_value=p['min_number_of_filters'], max_value=p['max_numbe_of_filters'], step=16),
                 kernel_size = hp.Choice(f"conv_1_kernel_size_{i}",values=p['kernel_size']),
                 kernel_regularizer = tf.keras.regularizers.L2(l2=hp.Float(f"lr_{i}", min_value=1e-4, max_value=1e-2, sampling="log")),
                 activation=hp.Choice(f"activation_{i}", ["relu", "tanh"])   
                )
        )

    model.add(layers.Flatten())

    for j in range(i,hp.Int("num_layers",p['min_number_of_layers']+i,p['max_number_of_layers'])+1+i):
       
        model.add(
            layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"dense_units_{i}", min_value=p['min_units_per_layers'], max_value=p['max_units_per_layers'], step=32),
                kernel_regularizer = tf.keras.regularizers.L2(l2=hp.Float(f"dense_lr_{i}", min_value=1e-4, max_value=1e-2, sampling="log")),
                activation=hp.Choice(f"dense_layer_activation_{i}", ["relu", "tanh"]),
            )
        )
        if hp.Boolean(f"dense_dropout_{j+i}"):
            model.add(layers.Dropout(rate=0.25))



    return model

