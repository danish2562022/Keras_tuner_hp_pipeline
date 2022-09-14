from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers


    

p = {
        "min_number_of_layers": 1,
        "max_number_of_layers": 5,
        "model_type" : "c",
        "min_units_per_layers" : 32,
        "max_units_per_layers" : 512,
        "num_of_classes" : 10,
        "choose_optimizer": "adam",
        
    }


def build_model(self,hp):
   
    model = keras.Sequential()
    if p['model_type'] == 'c':
        model.add(layers.Flatten())
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

