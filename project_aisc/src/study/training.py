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

model = DeepSet(DataProcessor=atom_data,latent_dim = 1)

model.build_model()
model.phi.summary()
model.rho.summary()

callbacks = []
model.fit_model(X,Y,X_val,Y_val,callbacks= callbacks)
model.evaluate_model(X_test,Y_test)
model.visual_model_perform()

#display and save the prediction vs the observed value or the critical Temperature

observed_vs_predicted = pd.DataFrame({'Oberved Critical Temperature (K)':Y_test,'Predicted Critical Temperature (K)':np.array(model.rho.predict(X_test)).reshape(Y_test.shape[0],)})

sns_plot = sns.scatterplot(x = observed_vs_predicted['Oberved Critical Temperature (K)'],y= observed_vs_predicted['Predicted Critical Temperature (K)']).get_figure()

sns_plot.savefig("training_img/pred_vs_ob.png")

#%%
#Create and save the mono dimensional rapresentations of the molecules
mono_rapp = model.phi.predict(list(atom_data.dataset))
mono_temp = model.rho.predict(list(atom_data.dataset))
mono_temp.shape
mono_rapp.shape

mono_dataset = pd.DataFrame.from_dict({'x':np.reshape(mono_rapp,(mono_rapp.shape[0])),'temp_pred':np.reshape(mono_temp,(mono_rapp.shape[0])),'temp_oss': atom_data.t_c})
mono_dataset.to_csv('mono_dim_data/mono_dim_rapp.csv')
mono_dataset = pd.read_csv('mono_dim_data/mono_dim_rapp.csv',index_col=0)
mono_dataset.head()

#%%
#Plot the learned feature of the molecules vs the observed Temperature
plot_fig(mono_dataset.x,mono_dataset.temp_oss,color='ro',xlabel = 'x',ylabel='Observed Temperature(K)',save = False)

#%%
#Plot the learned feature of the molecules vs the Pred Temperature
plot_fig(mono_dataset.x,mono_dataset.temp_pred,color='bo',xlabel = 'x',ylabel='Observed Temperature(K)',save = False)

#%%
#Plot the Histogram of the molecules' feature
sns_hist =sns.histplot(mono_dataset.x).get_figure()

sns_hist.savefig('mono_dim_data/hist_mono_png')
#%%
sns_hist = sns.histplot(mono_dataset.x,kde = True).get_figure()

sns_hist.savefig('mono_dim_data/hist_mono_kde.png')

#%%
#Create and save the bi-dimensional rapresentation

bi_rapp = model.phi.predict(list(atom_data.dataset))
bi_temp = model.rho.predict(list(atom_data.dataset))

bi_dataset = pd.DataFrame.from_dict({'x0':np.moveaxis(bi_rapp,0,1)[0],'x1':np.moveaxis(bi_rapp,0,1)[1],'temp_pred':np.reshape(bi_temp,(bi_rapp.shape[0])),'temp_oss': atom_data.t_c})
bi_dataset.to_csv('bi_dim_data/bi_dim_rapp.csv')
bi_dataset = pd.read_csv('bi_dim_data/bi_dim_rapp.csv',index_col=0)
bi_dataset.head()
#Plot the features space with the temperature

sns_plot = sns.scatterplot(x = 'x0',y= 'x1',hue='temp_oss', data = bi_dataset).get_figure()

sns_plot.savefig("bi_dim_data/bi_dim_temp_rapp.png")
#%%
#Plot the projection of the rapp on one axis vs the Observed Temperature
plot_fig(bi_dataset.x0,bi_dataset.temp_oss,xlabel = 'x0',ylabel='Observed Temperature(K)',save = False)

#%%
#Plot the other projection of the rapp on one axis vs the Observed Temperature
plot_fig(bi_dataset.x1,bi_dataset.temp_oss,xlabel = 'x1',ylabel='Observed Temperature(K)',save = False)


#%%
#Create and save the tri-dimensional rappresentations of molecules
tri_rapp = model.phi.predict(list(atom_data.dataset))
tri_temp = model.rho.predict(list(atom_data.dataset))

tri_dataset = pd.DataFrame.from_dict({'x0':np.moveaxis(tri_rapp,0,1)[0],'x1':np.moveaxis(tri_rapp,0,1)[1],'x2':np.moveaxis(tri_rapp,0,1)[2],'temp_pred':np.reshape(tri_temp,(tri_rapp.shape[0])),'temp_oss': atom_data.t_c})
tri_dataset.to_csv('tri_dim_data/tri_dim_rapp.csv')
tri_dataset = pd.read_csv('tri_dim_data/tri_dim_rapp.csv',index_col=0)


plot_fig(tri_dataset.x0,tri_dataset.temp_oss,xlabel = 'x0',ylabel='Observed Temperature(K)',save = False,path = 'tri_dim_data/', name= 'x0_vs_ob_temp.png')
plot_fig(tri_dataset.x1,tri_dataset.temp_oss,xlabel = 'x1',ylabel='Observed Temperature(K)',save = False,path = 'tri_dim_data/', name= 'x1_vs_ob_temp.png')
plot_fig(tri_dataset.x2,tri_dataset.temp_oss,xlabel = 'x2',ylabel='Observed Temperature(K)',save = False,path = 'tri_dim_data/', name= 'x2_vs_ob_temp.png')

#%%
def plot_fig(x,y,color='ro',save = False,path= None,name = None,xlabel=None,ylabel = None):
    plt.plot(x,y,color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save:
        plt.savefig(path+name)
    plt.show()
