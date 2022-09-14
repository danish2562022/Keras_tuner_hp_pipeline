import re
import os
import glob
import numpy as np
import random
import time
configfile: "config.yaml"

singularity: 'lolcow.sif'

rule run_model:
    output: 'hello_1.txt'
    run:
        shell('echo world > {output}')
        import keras_tuner
        import tensorflow as tf
        print(tf.__version__)
        