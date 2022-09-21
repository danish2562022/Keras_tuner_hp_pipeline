import re
import os
import glob
import numpy as np
import random
import time
configfile: "config.yaml"
experiment_name = config['experiment_name']
model_name = config['input_files']['models'].split("/")[-1].split(".")[0]
path = os.path.join("best_model",experiment_name,model_name)
best_model_text = os.path.join(path,'best_param.txt')
rule run_model:
    input:
        app = config['input_files']['app'],
    output:
        best_model_param = str(best_model_text)
    run:
        import subprocess
        import pandas
        import os
        import keras_tuner
        import tensorflow as tf
        model_import = "from " + ".".join(config['input_files']['models'].split('/')[0:-1])+"."+config['input_files']['models'].split("/")[-1].split(".")[0] + " import *"
        physical_devices = tf.config.list_physical_devices('GPU')
        print("GPUs: ", physical_devices)
        epochs = config['training_config']['epochs']
        max_trials = config['training_config']['max_trials']
        python_interpreter = "python"
        
        command = f"{python_interpreter} {input.app} --epochs {epochs} --max_trials {max_trials} " \
                  f"--import_model_dataloader '{model_import}' --best_model_path '{path}' --best_model_param '{output.best_model_param}'"  
        print(command)
        os.system(command)



