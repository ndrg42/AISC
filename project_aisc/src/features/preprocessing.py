import pandas as pd

from data import make_dataset
from features import build_features
import yaml
from yaml import Loader
import pathlib
import numpy as np


def preprocess_data(problem, supercon_data, garbagein_data, test_split, val_split, seed):
    # Load atomic data
    ptable = make_dataset.get_periodictable()
    # Initialize the processor for atomic data
    atom_processor = build_features.AtomData(ptable)
    # Process atomic data
    atom_processed = atom_processor.get_atom_data()

    if problem == 'regression':
        sc_dataframe = make_dataset.get_supercon(name=supercon_data)
        tc = sc_dataframe['critical_temp']
    elif problem == 'classification':
        supercon_dataframe = make_dataset.get_supercon(name=supercon_data)
        garbagein_dataframe = make_dataset.get_supercon(name=garbagein_data)
        tc_supercon = supercon_dataframe['critical_temp']
        tc_garbage = garbagein_dataframe['critical_temp']
        # Merge supercondutors data non-superconductors data
        sc_dataframe = pd.concat([supercon_dataframe, garbagein_data], ignore_index=True)
        tc = pd.concat([tc_supercon,tc_garbage], ignore_index=True)

    # Initialize processor for SuperCon data
    supercon_processor = build_features.SuperConData(atom_processed, sc_dataframe, padding=10)
    # Process SuperCon data
    supercon_processed = supercon_processor.get_dataset()

    X, X_test, Y, Y_test = build_features.train_test_split(supercon_processed, tc, test_split, seed)
    X, X_val, Y, Y_val = build_features.train_test_split(X, Y, val_split, seed)

    np.save(str(pathlib.Path(__file__).absolute().parent.parent.parent) + '/data/processed/train/X_train.npy',
            np.array(X))
    np.save(str(pathlib.Path(__file__).absolute().parent.parent.parent) + '/data/processed/train/Y_train.npy',
            np.array(Y))
    np.save(str(pathlib.Path(__file__).absolute().parent.parent.parent) + '/data/processed/val/X_val.npy',
            np.array(X_val))
    np.save(str(pathlib.Path(__file__).absolute().parent.parent.parent) + '/data/processed/val/Y_val.npy',
            np.array(Y_val))
    np.save(str(pathlib.Path(__file__).absolute().parent.parent.parent) + '/data/processed/test/X_test.npy',
            np.array(X_test))
    np.save(str(pathlib.Path(__file__).absolute().parent.parent.parent) + '/data/processed/test/Y_test.npy',
            np.array(Y_test))


if __name__ == '__main__':
    preprocessing_config_path = str(pathlib.Path(__file__).absolute().parent.parent.parent) +\
                                '/config/preprocessing.yaml'

    with open(preprocessing_config_path) as file:
        preprocessing_config = yaml.load(file, Loader)

    preprocess_data(**preprocessing_config)
