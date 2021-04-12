import sys
sys.path.append('../data')
sys.path.append('../features')
sys.path.append('../model')
import DataLoader
import Processing
from Processing import DataProcessor
import DeepSets
from DeepSets import DeepSet
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%
#Load and prepare the data for the model traning
ptable = DataLoader.PeriodicTable()
sc_dataframe = DataLoader.SuperCon(sc_path = '../../data/raw/unique_m.csv')

atom_data = Processing.DataProcessor(ptable, sc_dataframe)


path = '../../data/processed/'
atom_data.load_data_processed(path + 'dataset_elaborated.csv')
atom_data.load_data_processed(path + 'dataset_label_elaborated.csv')
atom_data.build_Atom()
atom_data.build_dataset()

X,X_val,Y,Y_val = atom_data.train_test_split(atom_data.dataset,np.array(atom_data.t_c),test_size = 0.2)
X,X_test,Y,Y_test = atom_data.train_test_split(X,Y,test_size = 0.2)

#%%
#Build and train the deep set model
import importlib
importlib.reload(DeepSets)
from DeepSets import DeepSet
model = DeepSet(DataProcessor=atom_data,latent_dim = 1)

model.get_best_model(X,Y,X_val,Y_val)

model.load_best_model(directory ='../../models/best_model_11-04/',project_name= 'model_11-04-3')

model.evaluate_model(X_test,Y_test)
true_positive,true_negative,false_positive,false_negative = model.naive_classificator(10,X_test,Y_test)
model.confusion_matrix(X_test,Y_test)
