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
if not os.path.exists(path):
    os.makedirs(path)
rule run_model:
    