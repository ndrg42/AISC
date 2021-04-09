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

# m = model.get_best_model(X=X,Y=Y,X_val = X_val,Y_val =Y_val)
#m = model.rho_builder()
model.build_model()
model.phi.summary()
model.rho.summary()

callbacks = []
model.fit_model(X,Y,X_val,Y_val,callbacks= callbacks)
model.evaluate_model(X_test,Y_test)
model.visual_model_perform()
path_to_save = '../../models/'
model.save_model(path_to_save,'model0')

model.evaluate_model(X_test,Y_test)

#display and save the prediction vs the observed value or the critical Temperature

observed_vs_predicted = pd.DataFrame({'Oberved Critical Temperature (K)':Y_test,'Predicted Critical Temperature (K)':np.array(m.predict(X_test)).reshape(Y_test.shape[0],)})

sns_plot = sns.scatterplot(x = observed_vs_predicted['Oberved Critical Temperature (K)'],y= observed_vs_predicted['Predicted Critical Temperature (K)']).get_figure()

sns_plot.savefig("training_img/pred_vs_ob.png")

#%%
#Create and save the mono dimensional rapresentations of the molecules
mono_rapp = m.predict(list(atom_data.dataset))
mono_rapp = models[0].layers[10](list(atom_data.dataset))
mono_temp = m.predict(list(atom_data.dataset))
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
    plt.xlim([0.25,1.5])
    plt.show()
#%%

def model_builder(hp):
        model = builder_rho()
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-4,1e-5])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = hp_learning_rate),
                          loss='mean_squared_error',
                          metrics=['mean_absolute_error'])

        return model



def rho_builder(hp):

    phi = phi_builder(hp)
    inputs= [Input(33) for i in range(10)]
    outputs = [phi(i) for i in inputs]

    y = Add()(outputs)
    layers = hp.Int('units_11',min_value = 1,max_value = 5,step = 1)
    for layer in range(layers):
        y = Dense(hp.Int('units_'+str(11+layer),min_value = 32,max_value = 400,step = 32),activation='relu')(y)


    output = Dense(1,activation = "linear",activity_regularizer = 'l1')(y)
    rho = Model(inputs = inputs,outputs = output)

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4,1e-5,1e-6])

    rho.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = hp_learning_rate),
                      loss='mean_squared_error',
                      metrics=['mean_absolute_error'])

    return rho

# def get_best_model(X,Y,X_val,Y_val):
#
#     tuner = kt.Hyperband(rho_builder,
#                  objective='val_accuracy',
#                  max_epochs=5
#                  )
#     stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
#
#     tuner.search(X,Y, epochs=5, validation_data=(X_val,Y_val), callbacks=[stop_early])
#
#     model = tuner.get_best_models(num_models=1)
#
#     return model

import kerastuner as kt
from kerastuner.tuners import Hyperband

model = get_best_model(X,Y,X_val,Y_val)
from tensorflow.keras.layers import Dense,Input,Add
from tensorflow.keras.models import Model
import tensorflow as tf
tuner = kt.Hyperband(rho_builder,
                 objective='val_mean_absolute_error',
                 max_epochs=10
                 )

def phi_builder(hp):
    input_atom = Input(shape = (33))
    x = Dense(33,kernel_initializer=tf.keras.initializers.Identity(),use_bias=False,activation='linear')(input_atom)
    layers = hp.Int('units',min_value = 1,max_value = 10,step = 1)
    for layer in range(layers):
        x = Dense(hp.Int('units_'+str(layer),min_value = 32,max_value = 400,step = 32),activation='relu')(x)

    y = Dense(1,activation='linear')(x)
    phi = Model(inputs = input_atom,outputs = y)

    return phi

hp_learning_rate = hp.Choice('learning_rate', values=[1e-4,1e-5,1e-6])

phi.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = hp_learning_rate),
                      loss='mean_squared_error',
                      metrics=['mean_absolute_error'])



tuner.search(X, Y,
             epochs=10,
             validation_data=(X_val, Y_val))

models = tuner.get_best_models(num_models=2)
models[0].summary()
np.sqrt(models[1].evaluate(X_test,Y_test)[0])
path_to_save = '../../models/'
models[0].save(path_to_save,'model_best_0')
models[0].summary()
models[0].layers[10](X_test)
m = tf.keras.models.load_model(path_to_save)

np.sqrt(m.evaluate(X_test,Y_test,verbose=0)[0])
m.summary()
m.functional_1.summary()
rmse = []
for i in range(10):
    X,X_val,Y,Y_val = atom_data.train_test_split(atom_data.dataset,np.array(atom_data.t_c),test_size = 0.2)
    X,X_test,Y,Y_test = atom_data.train_test_split(X,Y,test_size = 0.2)
    rmse.append(np.sqrt(m.evaluate(X_test,Y_test,verbose=0)[0]))

np.array(rmse).mean()
np.array(rmse).std()
