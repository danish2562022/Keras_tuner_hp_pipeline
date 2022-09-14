
from cgi import print_environ
import arg_parse
import yaml
import time
import keras_tuner
from custom_model_tuning import *
from datasets.data_loader_images import *
from best_model import *
import argparse


st = time.time()

parser = argparse.ArgumentParser(description='Train my model.')
parser.add_argument('--epochs',
                      type =int,
                      default = 2,
                      help = "Number of epochs")
parser.add_argument('--max_trials',
                     type = int,
                     default = 2,
                     help = "Number of search spaces")
args = parser.parse_args()

with open('config.yaml') as f:
    config_file = yaml.load(f, Loader = yaml.FullLoader) 
model_name = config_file['input_files']['models'].split("/")[-1].split(".")[0]

tuner = keras_tuner.RandomSearch(
        hypermodel = CustomTuning(),
        max_trials = args.max_trials,
        overwrite = True,
        directory = "results",
        distribution_strategy=tf.distribute.MirroredStrategy(),
        project_name= str(model_name)+"_results",
    )

x_train,x_test,x_val,y_train,y_test,y_val = load_data() 
tuner.search(x_train, y_train, epochs=args.epochs, validation_data=(x_val, y_val))
best_hps = tuner.get_best_hyperparameters()[0]
with open("best_model_params.txt", "w") as external_file:
    print(best_hps.values, file = external_file)
    external_file.close()
# best_model()
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')


