import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from data import make_dataset
from features import build_features
from model import build_models
from utils.utils import save_results
import tensorflow as tf
import argparse
import yaml
from yaml import Loader
import pathlib
import sys


def train_parser():
    with open(str(pathlib.Path(__file__).absolute().parent.parent.parent)+'/config/available_model_config.yaml') as file:
        model_config = yaml.load(file, Loader)

    my_parser = argparse.ArgumentParser(prog='train model',
                                        description="train model on superconductivity data",
                                        usage='%(prog)s [options]',
                                        )
    my_parser.add_argument('-model',
                           action='store',
                           nargs='+',
                           metavar='MODEL',
                           help="""Select the model. Possible choice:
                                  - """ + '\n-'.join(model_config.keys())
                           )

    my_parser.add_argument('-config',
                           action='store',
                           nargs=1,
                           metavar='CONFIG',
                           help="""Use a custom config for the ML model.
                                   The model need to be specified"""
                           )

    my_parser.add_argument('-save',
                           action='store',
                           nargs='+',
                           help="Save the results into a folder ",
                           choices=['model', 'score', 'test', 'all'],
                           )

    # Parse the args
    args = my_parser.parse_args()

    return args


def main():
    # Load atomic data
    ptable = make_dataset.get_periodictable()
    # Initialize the processor for atomic data
    atom_processor = build_features.AtomData(ptable)
    # Process atomic data
    atom_processed = atom_processor.get_atom_data()

    len_argv = len(sys.argv)
    model_config = None
    model_config_path = None

    # Check if any argument is passed from cli
    if len_argv > 1:
        args = train_parser()
        # If ars passed don't contain model we set a default one (regressor)
        if args.model is not None:
            model_name = ' '.join(args.model)
        else:
            model_name = 'regressor'
        model_config_path = args.config

    else:
        model_name = 'regressor'
        args = None

    # We keep pd.Series because it keeps track of the index in the test
    if 'regressor' in model_name:
        # Load SuperCon dataset
        sc_dataframe = make_dataset.get_supercon(name='supercon.csv')
        tc = sc_dataframe['critical_temp']

    elif 'classifier' in model_name:
        # Load SuperCon dataset
        sc_dataframe = make_dataset.get_supercon(name='supercon_garbage_50k.csv')
        tc = sc_dataframe['critical_temp'].apply(lambda x: int(x > 0))

    # If a custom model is passed through the config file we load it with yaml
    if model_config_path is not None:
        model_config_path = model_config_path[0]
    else:
        model_config_path = str(pathlib.Path(__file__).absolute().parent.parent.parent) + '/config/model_config.yaml'

    with open(model_config_path) as file:
        model_config = yaml.load(file, Loader)

    # Initialize processor for SuperCon data
    supercon_processor = build_features.SuperConData(atom_processed, sc_dataframe)
    # Process SuperCon data
    supercon_processed = supercon_processor.get_dataset()

    X, X_test, Y, Y_test = build_features.train_test_split(supercon_processed, tc,
                                                           model_config['train setup']['test split'])
    X, X_val, Y, Y_val = build_features.train_test_split(X, Y, model_config['train setup']['validation split'])

    # Define model and train it
    model = build_models.get_model(model_name=model_name, model_config=model_config)
    callbacks = [tf.keras.callbacks.EarlyStopping(**model_config['train setup']['early stopping setup'])]
    model.fit(X, Y, validation_data=(X_val, Y_val), callbacks=callbacks, **model_config['train setup']['fit setup'])

    # Save scores and metrics' name
    score = model.evaluate(X_test, Y_test, verbose=0)
    metrics_name = [metric.name for metric in model.metrics]
    # Print the metric and the relative score of the model
    for name, value in zip(metrics_name, score):
        print(name + ':', value)

    if args is not None and args.save is not None:
        save_results(score, model, [Y_test, model.predict(X_test)], args.save)


if __name__ == '__main__':
    main()
