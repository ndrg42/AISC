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
sc_dataframe = DataLoader.SuperCon(sc_path = '../../data/raw/supercon_tot.csv')
sc_dataframe

atom_data = Processing.DataProcessor(ptable, sc_dataframe)


path = '../../data/processed/'
atom_data.load_data_processed(path + 'dataset_elaborated.csv')
atom_data.load_data_processed(path + 'dataset_label_elaborated.csv')
atom_data.build_Atom()
atom_data.build_dataset()

X,X_val,Y,Y_val = atom_data.train_test_split(atom_data.dataset,np.array(atom_data.t_c),test_size = 0.2)
X,X_test,Y,Y_test = atom_data.train_test_split(X,Y,test_size = 0.2)

Y = (Y>0).astype(int)
Y_val = (Y_val>0).astype(int)
Y_test = (Y_test>0).astype(int)

#%%
#Build and train the deep set model
import importlib
importlib.reload(Processing)
from DeepSets import DeepSet
model = DeepSet(DataProcessor=atom_data,latent_dim = 1)

model.get_best_model(X,Y,X_val,Y_val)

model.load_best_architecture(directory ='../../models/best_model_08-05/',project_name= 'model_08-05-0')
callbacks = []
model.rho.evaluate(X_test,Y_test)

model.fit_model(X,Y,X_val,Y_val,callbacks = callbacks)
model.evaluate_model(X_test,Y_test)
true_positive,true_negative,false_positive,false_negative = model.naive_classificator(70,X_test,Y_test)
model.confusion_matrix(X_test,Y_test)


#%%
def plot_fig(x,y,color='ro',save = False,path= None,name = None,xlabel=None,ylabel = None):
    plt.plot(x,y,color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save:
        plt.savefig(path+name)
    # plt.xlim([0.4,1])
    plt.show()
