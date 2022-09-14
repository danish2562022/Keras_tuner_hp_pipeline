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
       return model
       
    def fit(self, hp, model, x_train, y_train,epochs, validation_data, callbacks=None, **kwargs):

        batch_size = hp.Int("batch_size", 32, 128, step=32, default=64)
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(
            batch_size
        )
        validation_data = tf.data.Dataset.from_tensor_slices(validation_data).batch(
            batch_size
        )

        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

        if p["choose_optimizer"] == 'adam':
            optim = keras.optimizers.Adam(learning_rate=learning_rate)

        elif p["choose_optimizer"] == 'sgd':
            optim = keras.optimizers.SGD(learning_rate=learning_rate)

        if p["model_type"] == 'r':
            loss_fn =  tf.keras.losses.MeanAbsoluteError()
        else:
            loss_fn = tf.keras.losses.CategoricalCrossentropy()

        epoch_loss_metric = keras.metrics.Mean()



        @tf.function
        def run_train_step(features,labels):

            with tf.GradientTape() as tape:
                pred_labels = model(features)
                loss = loss_fn(labels, pred_labels)

                if model.losses:
                    loss += tf.math.add_n(model.losses)

            gradients = tape.gradient(loss, model.trainable_variables)
            optim.apply_gradients(zip(gradients, model.trainable_variables))


        @tf.function
        def run_val_step(images, labels):
            logits = model(images)
            loss = loss_fn(labels, logits)
            # Update the metric.
            epoch_loss_metric.update_state(loss)

        for callback in callbacks:
            callback.model = model    
        best_epoch_loss = float("inf")

        for epoch in tqdm(range(epochs)):
            print(f"Epoch: {epoch}")

            for features,labels in train_ds:
                run_train_step(features,labels)

            for features,labels in validation_data:
                run_val_step(features,labels)

            epoch_loss = float(epoch_loss_metric.result().numpy())
            for callback in callbacks:
             
                callback.on_epoch_end(epoch, logs={"my_metric": epoch_loss})
            epoch_loss_metric.reset_states()
            print(f"Epoch loss: {epoch_loss}")
            best_epoch_loss = min(best_epoch_loss, epoch_loss)


        return best_epoch_loss






        






        

