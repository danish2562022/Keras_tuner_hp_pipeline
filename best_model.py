
import keras_tuner
import tensorflow as tf
import keras
import numpy as np
import json
import ast
from datasets.data_loader_classification import *
from keras import layers
from contextlib import redirect_stdout
from models.model_fc import p



def best_model():
    
    
    with open('best_model_params.txt') as f:
        lines = f.readlines()

    x = lines[0][0:-1]
    best_model_hs = ast.literal_eval(x)
    
    if p['model_type'] =='r':
        loss_fn =  tf.keras.losses.MeanAbsoluteError()
    else:
        loss_fn = tf.keras.losses.CategoricalCrossentropy()


    model = keras.Sequential()
    if p['model_type'] == 'c':
        model.add(layers.Flatten())

    for i in range(1,best_model_hs['num_layers']+1):
        model.add(
             layers.Dense(
                 units = best_model_hs[f"units_{i}"],
                 kernel_regularizer = tf.keras.regularizers.L2(l2 = best_model_hs[f"lr_{i}"]),
                 activation = best_model_hs[f"activation_{i}"]),)
        if best_model_hs[f"dropout_{i}"]:
            model.add(layers.Dropout(rate=0.25))
    
    learning_rate = best_model_hs['lr']
    if p['choose_optimizer'] == 'adam':
        optim = keras.optimizers.Adam(learning_rate=learning_rate)
    elif p['choose_optimizer'] == 'sgd':
        optim = keras.optimizers.SGD(learning_rate=learning_rate)
    if p['model_type'] =='r':
        model.add(layers.Dense(1))
        model.compile(loss='mean_absolute_error',
                optimizer=optim)
    else:
        model.add(layers.Dense(p['num_of_classes'], activation="softmax"))
        model.compile(
            optimizer=optim, loss="categorical_crossentropy", metrics=["accuracy"],
        )
    
    x_train,x_test,x_val,y_train,y_test,y_val = load_data() 
    if p['model_type']  == 'r':
        print("Regression Model")
        model.build((None, x_train.shape[-1]))
    else:
        print("Classification Model")
        model.build((None, *x_train.shape[-3:]))
     
    with open("best_model_params.txt", "a+") as f:
        with redirect_stdout(f):
            model.summary()
     
    print(model.summary())

if __name__ == "__main__":
    best_model()








