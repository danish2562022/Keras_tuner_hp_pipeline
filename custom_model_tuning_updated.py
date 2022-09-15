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

        return model
       
    def fit(self, hp, model, x_train, y_train,epochs, validation_data, callbacks=None, **kwargs):

        batch_size = hp.Int("batch_size", 32, 128, step=32, default=64)
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(
            batch_size
        )
        validation_data = tf.data.Dataset.from_tensor_slices(validation_data).batch(
            batch_size
        )

        

        

        if p["model_type"] == 'r':
            loss_fn =  tf.keras.losses.MeanAbsoluteError()
        
        else:
            loss_fn = tf.keras.losses.CategoricalCrossentropy()
            

        epoch_loss_metric = keras.metrics.Mean()
