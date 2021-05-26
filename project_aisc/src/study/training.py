import sys
sys.path.append('../../src/data')
sys.path.append('../../src/features')
sys.path.append('../../src/model')
import DataLoader
import Processing
from Processing import DataProcessor
import DeepSets
from DeepSets import DeepSet
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import importlib
importlib.reload(DeepSets)
#%%
#Load and prepare the data for the model traning
ptable = DataLoader.PeriodicTable()
sc_dataframe = DataLoader.SuperCon(sc_path ='../../data/raw/supercon_tot.csv')
sc_dataframe= sc_dataframe[sc_dataframe['critical_temp']>0]
sc_dataframe.shape
(sc_dataframe['critical_temp'] == 0).sum()
atom_data = Processing.DataProcessor(ptable, sc_dataframe)
sc_dataframe.shape
path = '../../data/processed/'
atom_data.load_data_processed(path + 'dataset_supercon.csv')
atom_data.load_data_processed(path + 'dataset_supercon_label.csv')
atom_data.build_Atom()
atom_data.build_dataset()

# atom_data.save_data_csv(path,'dataset_supercon_label.csv',tc=True)

tc_classification = np.where(atom_data.t_c > 0,1,0)
tc_classification

atom_data.dataset = [np.expand_dims(atom_data.dataset[i],axis=2) for i in range(len(atom_data.dataset))]

X,X_val,Y,Y_val = atom_data.train_test_split(atom_data.dataset,np.array(atom_data.t_c),test_size = 0.2)
X,X_val,Y,Y_val = atom_data.train_test_split(atom_data.dataset,tc_classification,test_size = 0.2)
X,X_test,Y,Y_test = atom_data.train_test_split(X,Y,test_size = 0.2)

#%%
#Build and train the deep set model

model = DeepSet(DataProcessor=atom_data,latent_dim = 300)

model.load_best_architecture(directory='../../models/best_model_17-05/',project_name='model_17-05-1')
model.load_model(path='../../models/',name = 'regressor')

#%%
import csv
n_cycles = 5
for count in range(n_cycles):


    X,X_val,y,y_val = atom_data.train_test_split(atom_data.dataset,np.array(atom_data.t_c),test_size = 0.2)
    X,X_test,y,y_test = atom_data.train_test_split(X,y,test_size = 0.2)

    model = DeepSet(DataProcessor = atom_data,latent_dim = 1)
    model.load_best_architecture(directory='../../models/best_model_17-05/',project_name='model_17-05-1')

    callbacks = []
    model.fit_model(X,y,X_val,y_val,callbacks= callbacks,epochs=400,patience=30)


    with open('regressor_score.csv', mode='a') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csv_writer.writerow([model.rho.evaluate(X_test,y_test)[0],model.rho.evaluate(X_test,y_test)[1],model.rho.evaluate(X_test,y_test)[2],r2_score(model.rho.predict(X_test),y_test)])
#%%
regressor_score = pd.read_csv('regressor_score.csv',names= ['MSE','RMSE','MAE','R_Square'])
regressor_score.mean()
regressor_score.std()
#%%
import seaborn as sns
import matplotlib.pyplot as plt
fig,ax = plt.subplots(2,2)
#ax.subplots_adjust(hspace=0.6, wspace=0.3)
fig.suptitle('Regression Metrics',fontsize=20)
fig.set_figwidth(10)
fig.set_figheight(10)
sns.boxplot(regressor_score['MSE'],ax=ax[0,0])
sns.boxplot(regressor_score['RMSE'],ax=ax[0,1])
sns.boxplot(regressor_score['MAE'],ax=ax[1,0])
sns.boxplot(regressor_score['R_Square'],ax=ax[1,1])
plt.savefig('Regressor_Metrics.png')
plt.show()

#%%
model.rho.summary()

phi = model.rho.layers[10]
phi.compile(loss='mse',optimizer = 'adam',metrics=['mae'])
phi.evaluate(X_test,Y_test)
model.build_hybrid_model()
model.phi.summary()


callbacks = []
model.fit_model(X,Y,X_val,Y_val,callbacks= callbacks,epochs=400,patience=20)
model.rho.evaluate(X_test,Y_test)[0]

from sklearn.metrics import r2_score
r2_score(model.rho.predict(X_test),Y_test)

y_pred = np.reshape(np.where(model.rho.predict(X_test)>0.5,1,0),(np.where(model.rho.predict(X_test)>0.5,1,0).shape[0],))

from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
precision_score(Y_test,y_pred)
recall_score(Y_test,y_pred)
f1_score(Y_test,y_pred)
accuracy_score(Y_test,y_pred)

import csv
with open('class_score.csv', mode='a') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    csv_writer.writerow([accuracy_score(Y_test,y_pred),precision_score(Y_test,y_pred),recall_score(Y_test,y_pred),f1_score(Y_test,y_pred)])


from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
#%%
cm = confusion_matrix(Y_test,y_pred)
disp = ConfusionMatrixDisplay(cm,display_labels=['non sc','sc'])
disp.plot().figure_.savefig('class_conf_matrix.png')
#%%
n_cycles = 4
for count in range(n_cycles):

    tc_classification = np.where(atom_data.t_c > 0,1,0)

    X,X_val,y,y_val = atom_data.train_test_split(atom_data.dataset,tc_classification,test_size = 0.2)
    X,X_test,y,y_test = atom_data.train_test_split(X,y,test_size = 0.2)

    model = DeepSet(DataProcessor = atom_data,latent_dim = 1)
    model.load_best_architecture(directory='../../models/best_model_08-05/',project_name='model_08-05-0')

    callbacks = []
    model.fit_model(X,y,X_val,y_val,callbacks= callbacks,epochs=100,patience=15)
    y_pred = np.reshape(np.where(model.rho.predict(X_test)>0.5,1,0),(np.where(model.rho.predict(X_test)>0.5,1,0).shape[0],))

    with open('class_score.csv', mode='a') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csv_writer.writerow([accuracy_score(y_test,y_pred),precision_score(y_test,y_pred),recall_score(y_test,y_pred),f1_score(y_test,y_pred)])
#%%

class_score = pd.read_csv('class_score.csv',names = ['accuracy','precision','recall','f1'])

class_score.mean()
class_score.std()
#%%
import seaborn as sns
import matplotlib.pyplot as plt
fig,ax = plt.subplots(2,2)
#ax.subplots_adjust(hspace=0.6, wspace=0.3)
fig.suptitle('Classification Metrics',fontsize=20)
fig.set_figwidth(10)
fig.set_figheight(10)
sns.boxplot(class_score['accuracy'],ax=ax[0,0])
sns.boxplot(class_score['precision'],ax=ax[0,1])
sns.boxplot(class_score['recall'],ax=ax[1,0])
sns.boxplot(class_score['f1'],ax=ax[1,1])
plt.savefig('Classification_Metrics.png')
plt.show()

#%%

model.evaluate_model(X_test,Y_test)
true_positive,true_negative,false_positive,false_negative = model.naive_classificator(0,X_test,Y_test)
model.confusion_matrix(X_test,Y_test)


model.visual_model_perform()
path_to_save = '../../models/'
model.save_model(path_to_save,'model0')
model.load_best_model(directory = '../../models/best_model_16-04/',project_name ='model_16-04-0')
model.load_model(path='../../models',name='/')
model.load_best_architecture(directory = '../../models/best_model_11-04/',project_name ='model_11-04-3')
model.rho.layers[10].predict(X_test)
model.rho.layers[10].summary()

phi.save(path_to_save + 'phi_model')
#display and save the prediction vs the observed value or the critical Temperature

observed_vs_predicted = pd.DataFrame({'Oberved Critical Temperature (K)':y_test,'Predicted Critical Temperature (K)':np.array(model.rho.predict(X_test)).reshape(Y_test.shape[0],)})
#%%
sns_plot = sns.scatterplot(x = observed_vs_predicted['Oberved Critical Temperature (K)'],y= observed_vs_predicted['Predicted Critical Temperature (K)']).get_figure()
# plt.scatter(x=0.05,y=model.rho.predict(quasi_crystall),color = 'r')
# plt.scatter(x=0.8,y=2.23,color = 'y')
# plt.xlim([0,5])
# plt.ylim([0,5])
plt.savefig('predicted_vs_observed_regressor.png')


#%%
sns_plot.savefig("training_img/pred_vs_ob.png")
#%%
for i in range(10):
    X,X_val,Y,Y_val = atom_data.train_test_split(atom_data.dataset,np.array(atom_data.t_c),test_size = 0.2)
    X,X_test,Y,Y_test = atom_data.train_test_split(X,Y,test_size = 0.2)

    model = DeepSet(DataProcessor=atom_data,latent_dim = 1)
    model.build_model()
    callbacks = []
    model.fit_model(X,Y,X_val,Y_val,callbacks= callbacks)

    mono_rapp = model.phi.predict(list(atom_data.dataset))
    mono_temp = model.rho.predict(list(atom_data.dataset))

    mono_dataset = pd.DataFrame.from_dict({'x':np.reshape(mono_rapp,(mono_rapp.shape[0])),'temp_pred':np.reshape(mono_temp,(mono_rapp.shape[0])),'temp_oss': atom_data.t_c})
    mono_dataset.to_csv('mono_dim_data/mono_dim_10/mono_dim_rapp_'+str(i)+'.csv')


#%%
#Create and save the mono dimensional rapresentations of the molecules
mono_rapp = model.phi.predict(list(atom_data.dataset))
mono_rapp = model.rho.layers[10].predict(list(atom_data.dataset))
mono_temp = model.rho.predict(list(atom_data.dataset))
mono_temp.shape
mono_rapp.shape
mono_temp[:]
#changed t_pred con mono_rapp
mono_dataset = pd.DataFrame.from_dict({'x':np.reshape(mono_rapp[:,0],(mono_rapp.shape[0])),'temp_pred':np.reshape(mono_temp[:,0],(mono_rapp.shape[0])),'temp_oss': atom_data.t_c})
mono_dataset.to_csv('mono_dim_data/mono_dim_rapp.csv')
mono_dataset = pd.read_csv('mono_dim_data/mono_dim_rapp.csv',index_col=0)
mono_dataset.head()

#%%
#Plot the learned feature of the molecules vs the observed Temperature
plot_fig(mono_dataset.x,mono_dataset.temp_oss,color='ro',xlabel = 'x',ylabel='Observed Temperature(K)',save = False)

#%%
#Plot the learned feature of the molecules vs the Pred Temperature
plot_fig(mono_dataset.x,mono_dataset.temp_pred,color='bo',xlabel = 'x',ylabel='Observed Temperature(K)',save =False)#True,path ='../../notebooks/',name='best_model_x1_vsx3.png')

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
    plt.xlim([-0.2,0.2])
    plt.show()
#%%
#Regression with quasi_crystall superconductors




#%%
#Classification with quasi_crystall superconductors



#%%
#Plot latent dim= 2 for atoms by regressor representation

#Split the dataset in train, validation and test set
X,X_val,Y,Y_val = atom_data.train_test_split(atom_data.dataset,np.array(atom_data.t_c),test_size = 0.2)
X,X_test,Y,Y_test = atom_data.train_test_split(X,Y,test_size = 0.2)

#Create the model with latent dim= 2 and train it
model = DeepSet(DataProcessor=atom_data,latent_dim = 2)

callbacks = []
model.fit_model(X,Y,X_val,Y_val,callbacks= callbacks,epochs=400,patience=20)

#Extrapolate the model phi that take as input atomic features and gives as output bidimensional atomic features in the latent space
phi = model.rho.layers[10]
phi.compile(loss = 'mse',optimizer = 'adam',metrics = ['mae'])

#Create the bidimensional representation of the atoms
atom_latent_space = phi.predict(atom_data.dataset)
#Plot the representation of the atomic latent space
sns.scatterplot(atom_latent_space[0],atom_latent_space[1])
#%%
#Plot latent dim= 2 for molecules by regressor representation


#%%
#Plot latent dim= 2 for atoms by Classifier representation

#%%
#Plot latent dim= 2 for molecules by Classifier representation

#%%
#cross-control for wrong prediction with regressor and Classifier

regressor = DeepSet(DataProcessor=atom_data,latent_dim = 300)
regressor.load_model(path='../../models/',name = 'regressor')

X_r,X_val_r,Y_r,Y_val_r = atom_data.train_test_split(atom_data.dataset,np.array(atom_data.t_c),test_size = 0.2)
X_r,X_test_r,Y_r,Y_test_r = atom_data.train_test_split(X_r,Y_r,test_size = 0.2)

callbacks = []
regressor.fit_model(X_r,Y_r,X_val_r,Y_val_r,callbacks= callbacks,epochs=400,patience=20)

classifier = DeepSet(DataProcessor=atom_data,latent_dim = 300)
classifier.load_model(path='../../models/',name = 'classifier')

tc_classification = np.where(atom_data.t_c > 0,1,0)

X_c,X_val_c,Y_c,Y_val_c = atom_data.train_test_split(atom_data.dataset,tc_classification,test_size = 0.2)
X_c,X_test_c,Y_c,Y_test_c = atom_data.train_test_split(X_c,Y_c,test_size = 0.2)

classifier.fit_model(X_c,Y_c,X_val_c,Y_val_c,callbacks= callbacks,epochs=400,patience=20)

Y_pred_r = regressor.predict(X_test_r)

threshold = 20

outliers = np.abs(Y_pred_r - Y_test_r) > threshold

material_outliers_r = atom_data.dataset['material'][outliers]


Y_pred_c = classifier.predict(X_test_c)

false_pred = (Y_pred_c == Y_test_c)
material_outliers_c = atom_data.dataset['material'][false_pred]

material_outliers_c == material_outliers_r



#%%
#Autoencoder
