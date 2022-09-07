
import keras_tuner
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import ast
from data_loader import *
from tensorflow.keras import layers
import arg_parse
from contextlib import redirect_stdout




def best_model():
    
    parser = arg_parse.get_args()
    args = parser.parse_args()
    with open('best_model_params.txt') as f:
        lines = f.readlines()

    x = lines[0][0:-1]
    best_model_hs = ast.literal_eval(x)
    
    if args.model_type =='r':
        loss_fn =  tf.keras.losses.MeanAbsoluteError()
    else:
        loss_fn = tf.keras.losses.CategoricalCrossentropy()


    model = keras.Sequential()
    if args.model_type == 'c':
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
    if args.choose_optimizer == 'adam':
        optim = keras.optimizers.Adam(learning_rate=learning_rate)
    elif args.choose_optimizer == 'sgd':
        optim = keras.optimizers.SGD(learning_rate=learning_rate)
    if args.model_type=='r':
        model.add(layers.Dense(1))
        model.compile(loss='mean_absolute_error',
                optimizer=optim)
    else:
        model.add(layers.Dense(args.num_of_classes, activation="softmax"))
        model.compile(
            optimizer=optim, loss="categorical_crossentropy", metrics=["accuracy"],
        )
    
    x_train,x_test,x_val,y_train,y_test,y_val = data_loader() 
    if args.model_type == 'r':
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








