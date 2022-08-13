import os

import pandas

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
import mlflow
import numpy as np


def train_parser():
    with open(str(pathlib.Path(
            __file__).absolute().parent.parent.parent) + '/config/available_model_config.yaml') as file:
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

    my_parser.add_argument('--no-save',
                           action='store',
                           nargs='?',
                           help="Don't save/track results with mlflow. Arguments specify what to not track",
                           choices=['model', 'all'],
                           const='all'
                           )

    # Parse the args
    args = my_parser.parse_args()

    return args


def main():
    len_argv = len(sys.argv)
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

        if args.no_save == 'all':
            disable_autolog = True
            log_models = False
        elif args.no_save == 'model':
            disable_autolog = False
            log_models = False
        else:
            disable_autolog = False
            log_models = True
    else:
        model_name = 'regressor'
        disable_autolog = False
        log_models = True

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

    mlflow.set_tracking_uri(str(pathlib.Path(__file__).absolute().parent.parent.parent) + "/data/experiments/mlruns")
    mlflow.set_experiment(model_name)

    # Load atomic data
    ptable = make_dataset.get_periodictable()
    # Initialize the processor for atomic data
    atom_processor = build_features.AtomData(ptable)
    # Process atomic data
    atom_processed = atom_processor.get_atom_data()

    # Initialize processor for SuperCon data
    supercon_processor = build_features.SuperConData(atom_processed, sc_dataframe)
    # Process SuperCon data
    supercon_processed = supercon_processor.get_dataset()

    X, X_test, Y, Y_test = build_features.train_test_split(supercon_processed, tc,
                                                           model_config['train setup']['test split'])
    X, X_val, Y, Y_val = build_features.train_test_split(X, Y, model_config['train setup']['validation split'])

    with mlflow.start_run():

        # Define model and train it
        model = build_models.get_model(model_name=model_name, model_config=model_config)
        callbacks = [tf.keras.callbacks.EarlyStopping(**model_config['train setup']['early stopping setup'])]

        # Logs metrics, params, model
        mlflow.tensorflow.autolog(disable=disable_autolog, log_models=log_models)

        model.fit(X, Y, validation_data=(X_val, Y_val), callbacks=callbacks, **model_config['train setup']['fit setup'])

        # Save scores and metrics' name
        score = model.evaluate(X_test, Y_test, verbose=0)
        metrics_name = [metric.name for metric in model.metrics]

        if log_models:
            # Save predictions of the model
            artifact_uri = mlflow.get_artifact_uri()
            np.save(artifact_uri + "/predictions.npy", tf.reshape(model.predict(X_test), shape=(-1,)).numpy())

    # Print metrics of the model
    for name, value in zip(metrics_name, score):
        print(name + ':', value)


if __name__ == '__main__':
    main()
