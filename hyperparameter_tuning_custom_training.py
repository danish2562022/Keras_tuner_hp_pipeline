
import arg_parse
import yaml
import time
import keras_tuner
from custom_model_tuning import *
import argparse


st = time.time()
with open('config.yaml') as f:
    config_file = yaml.load(f, Loader = yaml.FullLoader) 
model_name = config_file['input_files']['models'].split("/")[-1].split(".")[0]
program = "from models."+ model_name + " import build_model,p,load_data"
exec(program)
parser = argparse.ArgumentParser(description='Train my model.')
parser.add_argument('--epochs',
                      type =int,
                      default = 10,
                      help = "Number of epochs")
parser.add_argument('--max_trials',
                     type = int,
                     default = 10,
                     help = "Number of search spaces")
parser.add_argument('--best_model_path',
                     type = str,
                     default = None,
                     help = "give path to save best model")
args = parser.parse_args()

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
with open("best_model_params.txt", "w") as external_file:
    print(best_hps.values, file = external_file)
    external_file.close()
# best_model()
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
best_hp = tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(best_hp)
model.save("my_model")
model.summary()
