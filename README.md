# Keras_tuner_hyperparameter_optimization
Take Model as an input
# Hyperparameter Optimization Pipeline for MultiGPUs

Performance of machine learning model depends on the configuration of hyperparameters. Finding the optimal hyperparameters (hypeparameter tuning), could be very time consuming and challenging.  

In this project, we automate the hyperparameter optimization pipeline for classification and regression models.



## Getting Started

To know more about Keras Tuner, kindly refer official [docs](https://keras.io/api/keras_tuner/)


### Installing


Say what the step will be

    $ git clone [https://github.com/danish2562022/hyperparameter_optimization_pipeline.git](https://github.com/danish2562022/Keras_tuner_hyperparameter_optimization.git)
    $ cd Keras_tuner_hyperparameter_optimization
    $ conda env create -f environment.yml
    $ conda activate snakemake
   
    



## Running

Explain how to run the automated hyperparameter optimization pipeline

### Sample Tests

     $ snakemake --cores "all"
    
 Best model's hyperparameters get saved in best_model_params.txt
### Sample Output
    
    $ echo best_model/config[experiment_name]/config[input_files][models]/best_params.txt
    
    {'num_layers': 1, 'units_1': 480, 'lr_1': 0.001096082336813933, 'activation_1': 'relu', 'dropout_1': False, 'batch_size': 64, 'lr': 0.0001}

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten (Flatten)           (None, 784)               0         

     dense (Dense)               (None, 480)               376800    

     dense_1 (Dense)             (None, 10)                4810      

    =================================================================
    Total params: 381,610
    Trainable params: 381,610
    Non-trainable params: 0
    _________________________________________________________________



## Configuration file
Hyperparameter search space is defined in config_param.json
    
     {
     "min_number_of_layers": 1,
    "max_number_of_layers": 5,
    "model_type" : "c",
    "min_units_per_layers" : 32,
    "max_units_per_layers" : 512,
    "num_of_classes" : 10,
    "choose_optimizer": "adam",
    "epochs" : 10,
    "max_trials": 10
    }
        

## Built With

 

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code
of conduct, and the process for submitting pull requests to us.



## License



## Acknowledgments

  - Hat tip to anyone whose code is used
  - Inspiration
  - etc
