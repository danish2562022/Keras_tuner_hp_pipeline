import argparse
import json


def get_args():

    parser = argparse.ArgumentParser(
        description="Hyper parameter tuning of models.",
        usage="-------",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-c',
        '--config_file',
        dest='config_file',
        type=str,
        default=None,
        help='config file',
    )

    parser.add_argument(
                         '--config_files',
                        dest='config_files',
                        type=str,
                        default=None,
                        help='config file',)

    parser.add_argument('--max_number_of_layers',
                        action = 'store',
                        default = 5,
                        type = int,
                        help = "Minimum number of layers in architectures"
                        )

    parser.add_argument('--min_number_of_layers',
                        action = 'store',
                        default = 1,
                        type = int,
                        help = "Maximum number of layers in architectures"
                        )

    parser.add_argument('--epochs',
                        action = 'store',
                        default = 5,
                        type = int,
                        help = "Number of epochs"
                        )

    parser.add_argument('--max_trials',
                        action = 'store',
                        default = 50,
                        type = int,
                        help = "Number of epochs"
                        )

    parser.add_argument('--model_type',
                        action = 'store',
                        default = 'r',
                        type= str,
                        help= 'r for regression and c for classification')

    parser.add_argument('--min_units_per_layers',
                        action = 'store',
                        default = 32,
                        type= int,
                        help= 'minimum number of units per layers')

    parser.add_argument('--max_units_per_layers',
                        action = 'store',
                        default = 512,
                        type= int,
                        help= 'minimum number of units per layers')

    parser.add_argument('--num_of_classes',
                        action = 'store',
                        default = 10,
                        type= int,
                        help= 'Number of classes in classiffication model')

    parser.add_argument('--choose_optimizer',
                        action = 'store',
                        default = "adam",
                        type= str,
                        help= 'Choose optimizers: "adam" for Adam, "sgd" for SGD')




    args, unknown = parser.parse_known_args()
    parser_arg = argparse.ArgumentParser(parents=[parser], add_help=False)

    if args.config_files is not None:
        print(args.config_files)
        if '.json' in args.config_files:
         
            config = json.load(open(args.config_files))
            print(config)
            parser_arg.set_defaults(**config)

            [
                parser_arg.add_argument(arg)
                for arg in [arg for arg in unknown if arg.startswith('--')]
                if arg.split('--')[-1] in config
            ]

    print(parser_arg.parse_args())
    parser_arg.parse_args()
    return parser_arg



   


