from data import make_dataset
from features import build_features
import yaml
from yaml import Loader
import pathlib
import numpy as np

# Load atomic data
ptable = make_dataset.get_periodictable()
# Initialize the processor for atomic data
atom_processor = build_features.AtomData(ptable)
# Process atomic data
atom_processed = atom_processor.get_atom_data()

sc_dataframe = make_dataset.get_supercon(name='supercon.csv')
tc = sc_dataframe['critical_temp']

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


np.save(str(pathlib.Path(__file__).absolute().parent.parent.parent) + '/data/processed/train/X_train.npy', np.array(X))
np.save(str(pathlib.Path(__file__).absolute().parent.parent.parent) + '/data/processed/train/Y_train.npy', np.array(Y))
np.save(str(pathlib.Path(__file__).absolute().parent.parent.parent) + '/data/processed/val/X_val.npy', np.array(X_val))
np.save(str(pathlib.Path(__file__).absolute().parent.parent.parent) + '/data/processed/val/Y_val.npy', np.array(Y_val))
np.save(str(pathlib.Path(__file__).absolute().parent.parent.parent) + '/data/processed/test/X_test.npy', np.array(X_test))
np.save(str(pathlib.Path(__file__).absolute().parent.parent.parent) + '/data/processed/test/Y_test.npy', np.array(Y_test))