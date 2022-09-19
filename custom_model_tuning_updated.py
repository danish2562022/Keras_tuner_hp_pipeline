import keras_tuner
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from tqdm import tqdm
from models.model_cnn import build_model,p


class CustomTuning(keras_tuner.HyperModel):

    def build(self,hp):
        
        
        print(p)
        model = build_model(self,hp) 
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        if p["choose_optimizer"] == 'adam':
            optim = keras.optimizers.Adam(learning_rate=learning_rate)

        elif p["choose_optimizer"] == 'sgd':
            optim = keras.optimizers.SGD(learning_rate=learning_rate)
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"],
        )
        return model
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [16, 32]),
            **kwargs,
        )

        