import arg_parse
import keras_tuner
from model import *
from data_loader import *


parser = arg_parse.get_args()
args = parser.parse_args()
print(args.num_of_classes)
print(args.max_number_of_layers)
print(args.model_type)

if args.model_type =='c':
    tuner = keras_tuner.RandomSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=2,
        executions_per_trial=1,
        overwrite=True,
        directory="my_dir",
        project_name="helloworld",
    )

else:
    tuner = keras_tuner.RandomSearch(
        hypermodel=build_model,
        objective="mse",
        max_trials=2,
        executions_per_trial=1,
        overwrite=True,
        directory="my_dir",
        project_name="helloworld",
    )

x_train,x_test,x_val,y_train,y_test,y_val = data_loader() 
tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val))
models = tuner.get_best_models(num_models=2)
best_model = models[0]
best_model.build(input_shape=(None, 28, 28))
print(best_model.summary())
print(tuner.results_summary())

