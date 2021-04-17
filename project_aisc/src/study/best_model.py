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

np.array(atom_data.dataset).shape
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
importlib.reload(Processing)
from DeepSets import DeepSet
model = DeepSet(DataProcessor=atom_data,latent_dim = 1)

model.get_best_model(X,Y,X_val,Y_val)

model.load_best_model(directory ='../../models/best_model_16-04/',project_name= 'model_16-04-0')

model.evaluate_model(X_test,Y_test)
true_positive,true_negative,false_positive,false_negative = model.naive_classificator(70,X_test,Y_test)
model.confusion_matrix(X_test,Y_test)

model.rho.layers[10].predict()


mono_rapp = model.rho.layers[10].predict(list(atom_data.dataset))
mono_temp = model.rho.predict(list(atom_data.dataset))
mono_dataset = pd.DataFrame.from_dict({'x':np.reshape(mono_rapp[:,319],(mono_rapp.shape[0])),'temp_pred':np.reshape(mono_rapp[:,317],(mono_rapp.shape[0])),'temp_oss': atom_data.t_c})
plot_fig(mono_dataset.x,mono_dataset.temp_pred,color='ro',xlabel = 'x',ylabel='Observed Temperature(K)',save = False)
#%%
def plot_fig(x,y,color='ro',save = False,path= None,name = None,xlabel=None,ylabel = None):
    plt.plot(x,y,color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save:
        plt.savefig(path+name)
    # plt.xlim([0.4,1])
    plt.show()
#%%
#implement correlation function to estimate feature importance

model.rho.layers[10].summary()

from tensorflow.keras import backend as K

gx0x1 = K.function([model.rho.layers[10].layers[0].input],[model.rho.layers[10].layers[0].output])
gx0x1(list(atom_data.dataset))[0][:,0].shape
xy = mono_temp*np.reshape(gx0x1(list(atom_data.dataset))[0][:,2],mono_temp.shape)
xy.mean()-  x_med*y_med
x_med = np.reshape(gx0x1(list(atom_data.dataset))[0][:,0],mono_temp.shape).mean()
y_med = mono_temp.mean()
model.input_dim
corr_score = {}
for index in range(model.input_dim):

    gx0x1 = K.function([model.rho.layers[10].layers[0].input],[model.rho.layers[10].layers[0].output])

    xy = mono_temp*np.reshape(gx0x1(list(atom_data.dataset))[0][:,index],mono_temp.shape)

    x_med = np.reshape(gx0x1(list(atom_data.dataset))[0][:,0],mono_temp.shape).mean()
    y_med = mono_temp.mean()
    corr_score['G(x'+str(index)+',t_c)'] = xy.mean()-  x_med*y_med


corr_score
import importlib
importlib.reload(DataLoader)
atom_data.build_Atom()
atom_data.Atom
gx0x1(list(atom_data.dataset))[0][:,]

corr_score = {}
for index in range(model.input_dim):
    for k in range(10):
        gx0x1 = np.array(atom_data.dataset)[k,:,index]

        xy = mono_temp*np.reshape(gx0x1,mono_temp.shape)

        x_med = np.reshape(gx0x1,mono_temp.shape).mean()
        y_med = mono_temp.mean()
        try:
            corr_score['G(x'+str(index)+',t_c)'] += xy.mean()-  x_med*y_med
        except:
            corr_score['G(x'+str(index)+',t_c)'] = 0
            corr_score['G(x'+str(index)+',t_c)'] += xy.mean()-  x_med*y_med

len(corr_score.keys())
len(atom_data.Atom.columns)
x0 = list(atom_data.Atom.columns)
x0.append('%')
dict(zip(x0,list(corr_score.values())))
corr_score
