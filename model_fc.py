from tensorflow import keras
from tensorflow.keras import layers



def params():

    params = {
        "min_number_of_layers": 1,
        "max_number_of_layers": 5,
        "model_type" : "c",
        "min_units_per_layers" : 32,
        "max_units_per_layers" : 512,
        "num_of_classes" : 10,
        "choose_optimizer": "adam",
        "epochs" : 10,
        "max_trials": 2
    }
    return params
    


def build_model(hp,input_shape = (28, 28)):
    p = params()
    
    model = keras.Sequential()
    model.add(keras.Input(shape=(28, 28)))
    model.add(layers.Flatten())
    for i in range(hp.Int("num_layers",1,p['max_number_of_layers'])):
       
        model.add(
            layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=p['min_units_per_layers'], max_value=p['max_units_per_layers'], step=32),
                activation=hp.Choice("activation", ["relu", "tanh"]),
            )
        )
        if hp.Boolean(f"dropout_{i}"):
            model.add(layers.Dropout(rate=0.25))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

    if p['choose_optimizer'] == 'adam':
        optim = keras.optimizers.Adam(learning_rate=learning_rate)
    elif p['choose_optimizer'] == 'sgd':
        optim = keras.optimizers.SGD(learning_rate=learning_rate)
    
    if p['model_type'] == 'r':
        model.add(layers.Dense(1))
        model.compile(loss='mean_absolute_error',
                optimizer=optim)
    else:
        model.add(layers.Dense(p['num_of_classes'], activation="softmax"))
        model.compile(
            optimizer=optim, loss="categorical_crossentropy", metrics=["accuracy"],
        )
    return model

