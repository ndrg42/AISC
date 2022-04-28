import sys
sys.path.append('/home/claudio/AISC/project_aisc/src/data')
sys.path.append('/home/claudio/AISC/project_aisc/src/features')
sys.path.append('/home/claudio/AISC/project_aisc/src/model')
import make_dataset
import build_features
import build_models
import tensorflow as tf
import yaml
from yaml import Loader
import importlib
importlib.reload(build_models)
import os
os.getcwd()
os.chdir('/home/claudio/AISC/project_aisc/')
from tensorflow.keras.layers import Add,Multiply
import seaborn as sns
import numpy as np
from mendeleev import element
import pandas as pd
#%%
with open('/home/claudio/AISC/project_aisc/config/latent_dimension_config.yaml') as file:
    model_config = yaml.load(file,Loader)

#Load atomic data
ptable = make_dataset.PeriodicTable()
#Initialize the processor for atomic data
atom_processor = build_features.AtomData(ptable)
#Process atomic data
atom_processed = atom_processor.get_atom_data()

#Load SuperCon dataset
sc_dataframe = make_dataset.SuperCon(sc_path ='data/raw/supercon_tot.csv')
#Initialize processor for SuperCon data
supercon_processor = build_features.SuperConData(atom_processed,sc_dataframe,padding = 10)
#Process SuperCon data
supercon_processed = supercon_processor.get_dataset()

tc_regression = sc_dataframe['critical_temp']
X,X_test,Y,Y_test = build_features.train_test_split(supercon_processed, tc_regression, 0.2)
X,X_val,Y,Y_val = build_features.train_test_split(X, Y, 0.2)

build_models.model_config_initializer(model_config, model_name='linear regressor')
#Define model and train it
model = build_models.get_model(model_name = 'linear regressor',model_config = model_config)
callbacks = [tf.keras.callbacks.EarlyStopping(min_delta = 5, patience = 40, restore_best_weights = True)]
model.fit(X, Y, validation_data = (X_val,Y_val), epochs = 10, callbacks = callbacks)

materials_representation = model.material_representation(supercon_processed)

high_temperature_materials = np.where(tc_regression > 10,1,0)
sns.histplot(x = np.reshape(materials_representation.numpy(),(-1,)), hue = high_temperature_materials)

elements = [element(i).symbol for i in range(1,97)]

elements_dataframe = pd.DataFrame(np.eye(96),columns = elements)
atoms_processor = build_features.SuperConData(atom_processed,elements_dataframe)
#Process SuperCon data
atoms_processed = atoms_processor.get_dataset()

model.atom_representation(atoms_processed).numpy()
