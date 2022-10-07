import keras_tuner
import tensorflow as tf
import yaml
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from tqdm import tqdm
from matplotlib import pyplot as plt
from keras_tuner.engine import tuner_utils
      
with open('config.yaml') as f:
    config = yaml.load(f, Loader = yaml.FullLoader) 
program ="from " + ".".join(config['input_files']['models'].split('/')[0:-1])+"."+config['input_files']['models'].split("/")[-1].split(".")[0] + " import *"
exec(program)
class CustomTuning(keras_tuner.HyperModel):

    def build(self,hp):
           
        model = build_model(self,hp) 
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        if p["choose_optimizer"] == 'adam':
            optim = keras.optimizers.Adam(learning_rate=learning_rate)

        elif p["choose_optimizer"] == 'sgd':
            optim = keras.optimizers.SGD(learning_rate=learning_rate)
        
        if p["model_type"] == 'r':
            loss_fn =  tf.keras.losses.MeanAbsoluteError()
            model.compile(
                optimizer=optim, loss=loss_fn, metrics=["mean_squared_error"],
            )
        else:
            loss_fn = tf.keras.losses.CategoricalCrossentropy()
            
            model.compile(
                optimizer=optim, loss=loss_fn, metrics=["accuracy"],
            )
        return model
        
    
    def fit(self, hp, model, *args, **kwargs):
        
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", p['batch_size']),
            **kwargs,
        )
        