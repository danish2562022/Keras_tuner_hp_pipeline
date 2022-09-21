
from contextlib import redirect_stdout
import yaml
import time
import keras_tuner
from tensorflow import keras
from custom_model_tuning import *
import argparse
import os


st = time.time()
with open('config.yaml') as f:
    config_file = yaml.load(f, Loader = yaml.FullLoader) 

parser = argparse.ArgumentParser(description='Train my model.')
parser.add_argument('--epochs',
                      type =int,
                      default = 10,
                      help = "Number of epochs")
parser.add_argument('--max_trials',
                     type = int,
                     default = 10,
                     help = "Number of search spaces")
parser.add_argument('--import_model_dataloader',
                     type = str,
                     default = "from models.model_fc import *",
                     help = "import model and dataloader")
parser.add_argument('--best_model_path',
                     type = str,
                     default = "best_model/keras_tuner_fully_connected_pipeline/model_fc",
                     help = "give path to save best model")

parser.add_argument('--best_model_param',
                     type = str,
                     default = "best_model/keras_tuner_fully_connected_pipeline/model_fc/best_param.txt",
                     help = "give path to save best model parameters")
args = parser.parse_args()
model_name = args.import_model_dataloader.split(" ")[-3].split(".")[-1]
exec(args.import_model_dataloader)
obj = "val_accuracy" if p['model_type'] == "c" else "val_loss"
tuner = keras_tuner.RandomSearch(
        hypermodel = CustomTuning(),
        objective = obj,
        max_trials = args.max_trials,
        overwrite = True,
        directory = "results",
        distribution_strategy=tf.distribute.MirroredStrategy(),
        project_name= str(model_name)+"_results",
    )

x_train,x_test,x_val,y_train,y_test,y_val = load_data() 
tuner.search(x_train, y_train, epochs=args.epochs, validation_data=(x_val, y_val))
best_hps = tuner.get_best_hyperparameters()[0]


# best_model()
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
best_hp = tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(best_hp)
model.save(args.best_model_path)
rest_model = keras.models.load_model(args.best_model_path)
print("_"*100)
print("Best Hyperparameters value: ")
print(best_hps.values)
print("_"*100)
print("Model Summary: ")
rest_model.summary()
best_hps_save = os.path.join(args.best_model_param)
with open(best_hps_save, "w") as external_file:
    print(best_hps.values, file = external_file)
    with redirect_stdout(external_file):
        rest_model.summary()
    external_file.close()
