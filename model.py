from tensorflow import keras
from tensorflow.keras import layers


import arg_parse
parser = arg_parse.get_args()
opt = parser.parse_args()
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    for i in range(hp.Int("num_layers",1,opt.max_number_of_layers)):
       
        model.add(
            layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=opt.min_units_per_layers, max_value=opt.max_units_per_layers, step=32),
                activation=hp.Choice("activation", ["relu", "tanh"]),
            )
        )
        if hp.Boolean(f"dropout_{i}"):
            model.add(layers.Dropout(rate=0.25))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

    if opt.choose_optimizer == 'adam':
        optim = keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt.choose_optimizer == 'sgd':
        optim = keras.optimizers.SGD(learning_rate=learning_rate)
    
    if opt.model_type == 'r':
        model.add(layers.Dense(1))
        model.compile(loss='mean_absolute_error',
                optimizer=optim)
    else:
        model.add(layers.Dense(opt.num_of_classes, activation="softmax"))
        model.compile(
            optimizer=optim, loss="categorical_crossentropy", metrics=["accuracy"],
        )
    return model

