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


atom_data = Processing.DataProcessor(ptable, sc_dataframe)

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

model = DeepSet(DataProcessor=atom_data,latent_dim = 2)

model.load_best_architecture(directory='../../models/best_model_08-05/',project_name='model_08-05-0')
model.load_model(path='../../models/',name = 'classifier')
model.save_model(path='../../models/',name = 'classifier')

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
model.build_regressor()


callbacks = []
model.fit_model(X,Y,X_val,Y_val,callbacks= callbacks,epochs=10,patience=20)
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


#%%
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
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
model.rho.layers[11](X_test).shape
model.rho.layers[11].summary()

phi.save(path_to_save + 'phi_model')
#display and save the prediction vs the observed value or the critical Temperature

observed_vs_predicted = pd.DataFrame({'Oberved Critical Temperature (K)':Y_test,'Predicted Critical Temperature (K)':np.array(model.rho.predict(X_test)).reshape(Y_test.shape[0],)})
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
quasi_crystall_0 = 'Al14.9Mg44.1Zn41.0'
AC_0 = 'Al14.9Mg43.0Zn42.0'
qc = 'Al49.0Zn49.0Mg32'

#%%
#Classification with quasi_crystall superconductors
quasi_crystall_0 = 'Al14.9Mg44.1Zn41.0'
AC_0 = 'Al14.9Mg43.0Zn42.0'
qc = 'Al49.0Zn49.0Mg32'
quasi_cristalli = ['Al65Co15Cu20','Al70Co20Ni10','Al70Co15Ni15','Al72Co8Ni20','Al70.6Co6.7Ni22.7','Al78Mn22','Al70.5Mn16.5Pd13','Al70Mn17Pd13','Al75Os10Pd15','Al73Mn21Si6','Al6Cu1Li3','Al57Cu11Li32','Al63Cu25Fe12.5','Al62Cu25Fe12.5','Al62Cu25Ru13','Al68.7Mn9.6Pd21.7','Al70Mn10Pd20','Al71Mn8Pd21','Al70.5Mn8.5Pd21','Al73Re9Pd18','Ti41.5Zr41.5Ni17','Zn60Mg31Ho9','Cd5.7Yb']
sc_qc=[]
len(quasi_cristalli)
for k in range(len(quasi_cristalli)):
    sc_qc.append(model.rho.predict(atom_data.get_input(quasi_cristalli[k]),batch_size=1))

sc_qc = np.reshape(np.array(sc_qc),(len(quasi_cristalli),))
(sc_qc>0.5).sum()
quasi_cristalli[11]
model.rho.predict(atom_data.get_input(quasi_cristalli[10]))
model.rho.predict(atom_data.get_input(quasi_crystall_0))
model.rho.predict(atom_data.get_input(AC_0))
model.rho.predict(atom_data.get_input(qc))
#%%
#Plot latent dim= 2 for atoms by regressor representation

#Split the dataset in train, validation and test set
X,X_val,Y,Y_val = atom_data.train_test_split(atom_data.dataset,np.array(atom_data.t_c),test_size = 0.2)
X,X_test,Y,Y_test = atom_data.train_test_split(X,Y,test_size = 0.2)

#Create the model with latent dim= 2 and train it
model = DeepSet(DataProcessor=atom_data,latent_dim = 2)
model.rho.summary()
callbacks = []
model.build_regressor()
model.fit_model(X,Y,X_val,Y_val,callbacks= callbacks,epochs=4,patience=40,batch_size=1)
#model.save_model(path = '../../models/',name='bidimensional_linear_classifier')
np.sqrt(model.rho.evaluate(X_test,Y_test)[0])


#Extrapolate the model phi that take as input atomic features and gives as output bidimensional atomic features in the latent space
phi = model.rho.layers[10]
phi.compile(loss = 'mse',optimizer = 'adam',metrics = ['mae'])
phi.summary()
#Create the bidimensional representation of the atoms
atom_latent_space = []
from mendeleev import element
for i in range(1,87):
    symbol = element(i).symbol
    element_name = atom_data.get_input(symbol)
    atom_latent_space.append(phi.predict(element_name[0]))



element_name[0][:,32] = 1
atom_latent_space = np.reshape(np.array(atom_latent_space),(86,2))
#Plot the representation of the atomic latent space
bidimensional_atom_latent_space = pd.DataFrame({'X1':list(np.array(atom_latent_space[:,0])),'X2':list(np.array(atom_latent_space[:,1]))})
bidimensional_atom_latent_space.to_csv('bidimensional_atom_regression.csv',index=False)

element_name[0]
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow import set_random_seed


input = Input(shape=(33))
a = input[0][:-1]
input.shape
a.shape
c = tf.reshape(a,(tf.shape(input)[0],32))

o = input[0][-1:]
x = Dense(1,bias_initializer = 'zeros',kernel_initializer=tf.keras.initializers.RandomNormal(seed=42))(c)
#y = Dense(1,bias_initializer = 'zeros',kernel_initializer=tf.keras.initializers.RandomNormal(seed=42))(a)

x = tf.math.multiply(x,o)
model = Model(inputs=input,outputs=[x,o])
model.compile(loss='mse',optimizer='adam')
model.predict(element_name[0])

element_name[0][:,32] =100
[0][:,31]
0.08056

#%%
def build_phi():

    input_atom = Input(shape = (33))
    n_element = input_atom[0][-1:]
    c =  input_atom[0][:-1]
    c = tf.reshape(c,(tf.shape(input_atom)[0],32))
    #x = BatchNormalization()(input_atom)
    #x = Dropout(0.5)(x)
    x = Dense(300,activation = "relu")(c)
    #x = Dropout(0.5)(x)
    #x = BatchNormalization()(x)
    x = Dense(300,activation = "relu")(x)
    #x = Dropout(0.3)(x)
    x = Dense(300,activation = "relu")(x)
    #x = Dropout(0.3)(x)
    y = Dense(2,activation = "linear",activity_regularizer = 'l1')(x)
    y = tf.math.multiply(y,n_element)
    phi = Model(inputs = input_atom,outputs = y)
    phi.compile(loss='mse',optimizer='adam')
    return phi

prototype = build_phi()
r
prototype.predict(element_name[0])

element_name[0][:,32] =1000
#%%
plt.figure(figsize=(9,9))
ax = sns.scatterplot(atom_latent_space[:,0],atom_latent_space[:,1])
for line in range(0,atom_latent_space.shape[0]):
    plt.text(atom_latent_space[line,0]+0.2,atom_latent_space[line,1],element(line+1).symbol)
ax.set_title('Atomic Features (1-86) Predicted by Phi')
plt.plot([-3.5,-0.3,-0.2,0.2,0.3,3.5],[0,0,0,0,0,0],color='r')
plt.plot([0,0,0,0,0,0],[-3.5,-0.3,-0.2,0.2,0.3,3.5],color='r')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
plt.xlim([-3.5,3.5])
plt.ylim([-3.5,3.5])
#ax.get_figure().savefig('atomic_features_predicted_by_phi_with_symbol_saved_model_with_line.png')

#%%
#Plot latent dim= 2 for molecules by regressor representation

from tensorflow.keras.layers import Input,Add
from tensorflow.keras.models import Model
def build_phi_molecules():
    inputs= [Input(33) for i in range(10)]
    outputs = [phi(i) for i in inputs]

    y = Add()(outputs)

    model = Model(inputs = inputs,outputs=y)
    return model

phi_molecules = build_phi_molecules()
phi_molecules.compile(loss = 'mse',optimizer = 'adam',metrics = ['mae'])
phi_molecules.summary()
#Create the bidimensional representation of the molecules
molecules_latent_space = phi_molecules(atom_data.dataset)

bidimensional_latent_space = pd.DataFrame({'X1':list(np.array(molecules_latent_space[:,0])),'X2':list(np.array(molecules_latent_space[:,1])),'sc':atom_data.t_c})
bidimensional_latent_space.to_csv('bidimensional_molecules_regression.csv',index=False)
bidimensional_latent_space
#%%
#Plot the representation of the latent space of molecules
plt.figure(figsize=(7,6))
ax = sns.scatterplot(molecules_latent_space[:,0],molecules_latent_space[:,1],hue=np.array(atom_data.t_c))
ax.set_title('Features of the Molecules Predicted by Phi+Add')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
plt.plot([-3.5,-0.3,-0.2,0.2,0.3,3.5],[0,0,0,0,0,0],color='r')
plt.plot([0,0,0,0,0,0],[-3.5,-0.3,-0.2,0.2,0.3,3.5],color='r')
plt.xlim([-15,15])
plt.ylim([-15,15])
legend = ax.get_legend()
legend.set_title('critical temperature')
#ax.get_figure().savefig('features_molecules_predicted_by_phi_add_saved_model.png')

#%%
#Plot latent dim= 2 for atoms by Classifier representation

#Split the dataset in train, validation and test set
tc_classification = np.where(atom_data.t_c > 0,1,0)

X,X_val,Y,Y_val = atom_data.train_test_split(atom_data.dataset,tc_classification,test_size = 0.2)
X,X_test,Y,Y_test = atom_data.train_test_split(X,Y,test_size = 0.2)
model.phi.summary()
#Create the model with latent dim= 2 and train it
model = DeepSet(DataProcessor=atom_data,latent_dim = 2)
model.load_model(path = '../../models/',name='bidimensional_classifier')
model.build_classifier()
callbacks = []
model.fit_model(X,Y,X_val,Y_val,callbacks= callbacks,epochs=100,patience=10,batch_size = 1)
model.rho.evaluate(X_test,Y_test,batch_size = 1)
#Extrapolate the model phi that take as input atomic features and gives as output bidimensional atomic features in the latent space
phi = model.rho.layers[10]
phi.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
phi.summary()

#Create the bidimensional representation of the atoms

atom_latent_space = []
from mendeleev import element
for i in range(1,97):
    symbol = element(i).symbol
    element_name = atom_data.get_input(symbol)
    atom_latent_space.append(phi.predict(element_name[0]))

atom_latent_space = np.reshape(np.array(atom_latent_space),(96,2))
#Plot the representation of the atomic latent space
#%%
plt.figure(figsize=(10,10))
ax = sns.scatterplot(atom_latent_space[:,0],atom_latent_space[:,1])
for line in range(0,atom_latent_space.shape[0]):
    plt.text(atom_latent_space[line,0]+0.02,atom_latent_space[line,1],element(line+1).symbol)
#plt.plot(x,y,color='g')
#plt.plot([-0.4,-0.3,-0.2,0.2,0.3,0.4],[0,0,0,0,0,0],color='r')
#plt.plot([0,0,0,0,0,0],[-0.4,-0.3,-0.2,0.2,0.3,0.4],color='r')
#sns.scatterplot(atom_latent_space[95,0],atom_latent_space[95,1])
ax.set_title('Atomic Features (1-86) Predicted by Linear Phi Classifier')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
plt.xlim([-0.7,0.7])
plt.ylim([-0.7,0.7])
#ax.get_figure().savefig('atomic_features_predicted_by_phi_classifier_with_symbol_linear_phi_zoom.png')
#%%

bidimensional_atom_latent_space = pd.DataFrame({'X1':list(np.array(atom_latent_space[:,0])),'X2':list(np.array(atom_latent_space[:,1]))})
#bidimensional_atom_latent_space.to_csv('bidimensional_atom.csv',index=False)

#%%
#Plot latent dim= 2 for molecules by Classifier representation

from tensorflow.keras.layers import Input,Add
from tensorflow.keras.models import Model
def build_phi_classifier_molecules():
    inputs= [Input(33) for i in range(10)]
    outputs = [phi(i) for i in inputs]

    y = Add()(outputs)

    m = Model(inputs = inputs,outputs=y)
    return m

phi_molecules = build_phi_classifier_molecules()
phi_molecules.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
phi_molecules.summary()

molecules_latent_space
#Create the bidimensional representation of the molecules

molecules_latent_space = phi_molecules.predict(atom_data.dataset,batch_size = 1)
molecules_latent_space.shape
data = np.array(atom_data.dataset)
data[:,1,:].shape

qc_latent_space= []
qc_class =[]
for k in range(len(quasi_cristalli)):
    qc_latent_space.append(phi_molecules(atom_data.get_input(quasi_cristalli[k])))
    qc_class.append(np.reshape((model.rho.predict(atom_data.get_input(quasi_cristalli[2]))>0.5).astype('int'),(1,)))

qc_class = np.reshape(np.array(qc_class),(23))
qc_class
qc_latent_space = np.reshape(np.array(qc_latent_space),(23,2))
#%%
#Plot the representation of the latent space of molecules
plt.figure(figsize=(7,6))
ax = sns.scatterplot(molecules_latent_space[:,0],molecules_latent_space[:,1],hue=tc_classification)
plt.plot(x,y,color='g')
#plt.plot(x,y_ort,color='r')

#ax = sns.scatterplot(qc_latent_space[:,0],qc_latent_space[:,1],hue=qc_class,palette='bright')
ax.set_title('Features of the Molecules Predicted by linear Phi+Add')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
#plt.xlim([-4,4])
#plt.ylim([-4,4])
legend = ax.get_legend()
legend.set_title('critical temperature')
#ax.get_figure().savefig('features_molecules_predicted_by_linear_phi_add_classifier_logistic_regression.png')
#%%
plt.scatter(molecules_latent_space[:,0],tc_classification)
plt.plot(x,y)
model.load_model(path='../../models/',name = 'bidimensional_classifier')

bidimensional_latent_space = pd.DataFrame({'X1':list(np.array(molecules_latent_space[:,0])),'X2':list(np.array(molecules_latent_space[:,1])),'sc':tc_classification})
#bidimensional_latent_space.to_csv('bidimensional_molecules.csv',index=False)

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_m,X_m_test,Y_m,Y_m_test = train_test_split(bidimensional_latent_space[['X1','X2']],bidimensional_latent_space['sc'],test_size=0.2)
Y_m
logistic_regression = LogisticRegression().fit(X=X_m,y=Y_m)
logistic_regression.score(X=X_m_test,y=Y_m_test)

y_pred = logistic_regression.predict(X_m_test)

from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
precision_score(Y_m_test,y_pred)
recall_score(Y_m_test,y_pred)
f1_score(Y_m_test,y_pred)
accuracy_score(Y_m_test,y_pred)

b = logistic_regression.intercept_
w = logistic_regression.coef_

w = w/w[0][1]
b = b/w[0][1]
b[0]
w
x = np.arange(-25,30,0.1)
m = - w[0][0]/w[0][1]
y = m*x - b[0]/w[0][1]
y_ort = (-1/m) * x
#%%
#define ortogonal line to decision boundary of logistic_regression


def distance_from_decision_boundary(x,y):

    distance = (x*w[0][0]+y*w[0][1]+b[0])/np.sqrt(w[0][0]*w[0][0]+w[0][1]*w[0][1])

    return distance


#%%
np.sqrt(w[0][0]*w[0][0]+w[0][1]*w[0][1])
w[0][0]*w[0][0]
w[0][1]
#%%
distance = distance_from_decision_boundary(bidimensional_latent_space['X1'].values,bidimensional_latent_space['X2'].values)
plt.plot(x,y,color='g')
plt.scatter(bidimensional_latent_space['X1'].values[100],bidimensional_latent_space['X2'].values[100])
plt.scatter(bidimensional_latent_space['X1'].values[800],bidimensional_latent_space['X2'].values[800])
distance[800]
distance_pred = (distance>0).astype(int)
distance_pred.sum()
distance.shape
#%%
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0
bidimensional_latent_space.shape
for i in range(bidimensional_latent_space.shape[0]):
    if tc_classification[i] == 1:
        if distance_pred[i] == 1:
            true_positive +=1
        if distance_pred[i] == 0:
            false_negative +=1

    if tc_classification[i] == 0:
        if distance_pred[i] == 0:
            true_negative +=1
        if distance_pred[i] == 1:
            false_positive +=1

accuracy = (true_positive + true_negative)/(true_negative+true_positive+false_negative+false_positive)
precision = true_positive/(true_positive + false_positive)
recall = true_positive/(true_positive + false_negative)
f1= 2*(precision*recall/(precision + recall))
accuracy
precision
recall
f1
y_pred = distance_pred
Y_m_test= tc_classification

distance = distance_from_decision_boundary(X_m_test['X1'].values,X_m_test['X2'].values)

distance_pred = (distance>0).astype(int)

from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
precision_score(Y_m_test,y_pred)
recall_score(Y_m_test,y_pred)
f1_score(Y_m_test,y_pred)
accuracy_score(Y_m_test,y_pred)


#%%
#atom as distance from decision boundary
atom_distance = distance_from_decision_boundary(atom_latent_space[:,0],atom_latent_space[:,1])
atom_distance
atom_dict = {}
for k in range(1,87):
    atom_dict[element(k).symbol] = atom_distance[k-1]

atom_dict
from mendeleev import element

#sc if distance of the molecules>0 => sum distance atom/n-1 where n number of atoms > b
from DataLoader import from_string_to_dict
lista = []
from_string_to_dict(sc_dataframe['material'][100],lista)
mol_sum = 0
for j in range(len(lista)):
    mol_sum += atom_dict[lista[j][0]]*float(lista[j][1])

a_tilde = w[0][0]/np.sqrt(w[0][0]*w[0][0]+w[0][1]*w[0][1])
b_tilde = w[0][1]/np.sqrt(w[0][0]*w[0][0]+w[0][1]*w[0][1])
c_tilde = - b[0]/np.sqrt(w[0][0]*w[0][0]+w[0][1]*w[0][1])

for k in range(1,97):
    atom_dict[element(k).symbol] = list(atom_latent_space[k-1]) + [atom_latent_space[k-1][0]*a_tilde,atom_latent_space[k-1][1]*b_tilde,atom_latent_space[k-1][0]*a_tilde+atom_latent_space[k-1][1]*b_tilde]


dataset_atom = pd.DataFrame.from_dict(atom_dict,orient='index',columns = ['x1','x2','x1_tilde','x2_tilde','z_tilde'])
dataset_atom.to_csv('dataset_ai_atom.csv')
dataset_atom['z_tilde'].to_numpy()
c_tilde
atom_dict['H'][4]
distance_summed_mol = []
tc_classification_linear = []
for k in range(sc_dataframe['material'].shape[0]):
    lista = []
    from_string_to_dict(sc_dataframe['material'][k],lista)
    mol_sum = 0
    for j in range(len(lista)):
        mol_sum += atom_dict[lista[j][0]][4]*float(lista[j][1])

    tc_classification_linear.append((mol_sum > c_tilde).astype(int))
    distance_summed_mol.append(mol_sum-c_tilde)



len(tc_classification_linear)
sc_dataframe['material'].shape[0]
y_pred = np.array(tc_classification_linear)
Y_m_test = tc_classification
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
precision_score(Y_m_test,y_pred)
recall_score(Y_m_test,y_pred)
f1_score(Y_m_test,y_pred)
accuracy_score(Y_m_test,y_pred)
list_element = [i for i in range(1,atom_latent_space.shape[0]+1)]

#%%

plt.figure(figsize=(10,10))
ax = sns.scatterplot(list_element,dataset_atom['z_tilde'].to_numpy())
for line in range(0,len(list_element)):
    plt.text(list_element[line]+0.02,dataset_atom['z_tilde'].to_numpy()[line],dataset_atom.index[line])
#plt.plot(x,y,color='g')
#plt.plot([-0.4,-0.3,-0.2,0.2,0.3,0.4],[0,0,0,0,0,0],color='r')
plt.plot([-4,-0.3,-0.2,0.2,0.3,0.4,110],[0,0,0,0,0,0,0],color='r')
#sns.scatterplot(atom_latent_space[95,0],atom_latent_space[95,1])
ax.set_title('Atomic Features (1-96) Predicted by Linear Phi Classifier')
ax.set_xlabel('Element')
ax.set_ylabel('Z_tilde')
plt.xlim([0,110])
#plt.ylim([-0.7,0.7])
ax.get_figure().savefig('Z_tilde_vs_Element_with_line.png')

#%%
#cross-control for wrong prediction with regressor and Classifier

regressor = DeepSet(DataProcessor=atom_data,latent_dim = 2)
regressor.build_regressor()
regressor.load_model(path='../../models/',name = 'regressor')

X_r,X_val_r,Y_r,Y_val_r = atom_data.train_test_split(atom_data.dataset,np.array(atom_data.t_c),test_size = 0.2)
X_r,X_test_r,Y_r,Y_test_r = atom_data.train_test_split(X_r,Y_r,test_size = 0.2)

callbacks = []
regressor.fit_model(X_r,Y_r,X_val_r,Y_val_r,callbacks= callbacks,epochs=400,patience=40)

classifier = DeepSet(DataProcessor=atom_data,latent_dim = 2)
classifier.build_classifier()
classifier.load_model(path='../../models/',name = 'classifier')

tc_classification = np.where(atom_data.t_c > 0,1,0)

X_c,X_val_c,Y_c,Y_val_c = atom_data.train_test_split(atom_data.dataset,tc_classification,test_size = 0.2)
X_c,X_test_c,Y_c,Y_test_c = atom_data.train_test_split(X_c,Y_c,test_size = 0.2)

classifier.fit_model(X_c,Y_c,X_val_c,Y_val_c,callbacks= callbacks,epochs=400,patience=20)

Y_pred_r = np.reshape(regressor.rho.predict(atom_data.dataset),(atom_data.t_c.shape[0],))

threshold = 20

Y_pred_r.shape
atom_data.t_c.shape

outliers = np.abs(Y_pred_r - atom_data.t_c) > threshold
outliers
material_outliers_r = sc_dataframe['material'][outliers].values


Y_pred_c = np.reshape(classifier.rho.predict(atom_data.dataset),(atom_data.t_c.shape[0],))
Y_pred_c = np.where(Y_pred_c>0.5,1,0)

false_pred = (Y_pred_c != tc_classification)

material_outliers_c = sc_dataframe['material'][false_pred].values

material_outliers_c.shape
material_outliers_r.shape

common_wrong_pred = set(material_outliers_c).intersection(set(material_outliers_r))
common_wrong_pred = list(common_wrong_pred)
import csv
with open('common_wrong_prediction.csv', mode='a') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for k in range(len(common_wrong_pred)):
        csv_writer.writerow([common_wrong_pred[k]])
#%%
#outliers in latent space with regression model

regressor = DeepSet(DataProcessor=atom_data,latent_dim = 2)
regressor.build_regressor()


X,X_val,Y,Y_val = atom_data.train_test_split(atom_data.dataset,np.array(atom_data.t_c),test_size = 0.2)
X,X_test,Y,Y_test = atom_data.train_test_split(X,Y,test_size = 0.2)

callbacks = []
regressor.fit_model(X,Y,X_val,Y_val,callbacks= callbacks,epochs=400,patience=40)
regressor.evaluate_model(X_test,Y_test)
phi = regressor.rho.layers[10]
phi.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

phi_molecules = build_phi_molecules()
phi_molecules.compile(loss = 'mse',optimizer = 'adam',metrics = ['mae'])
phi_molecules.summary()
#Create the bidimensional representation of the molecules
molecules_latent_space = phi_molecules(atom_data.dataset)

#%%
quasi_cristalli = ['Al65Co15Cu20','Al70Co20Ni10','Al70Co15Ni15','Al72Co8Ni20','Al70.6Co6.7Ni22.7','Al78Mn22','Al70.5Mn16.5Pd13','Al70Mn17Pd13','Al75Os10Pd15','Al73Mn21Si6','Al6CuLi3','Al57Cu11Li32','Al63Cu25Fe12.5','Al62Cu25Fe12.5','Al62Cu25Ru13','Al68.7Mn9.6Pd21.7','Al70Mn10Pd20','Al71Mn8Pd21','Al70.5Mn8.5Pd21','Al73Re9Pd18','Ti41.5Zr41.5Ni17','Zn60Mg31Ho9','Cd5.7Yb']

#%%
#Plot the representation of the latent space of molecules
plt.figure(figsize=(7,6))
ax = sns.scatterplot(molecules_latent_space[:,0],molecules_latent_space[:,1],hue=atom_data.t_c)
ax.set_title('Features of the Molecules Predicted by Phi+Add')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
legend = ax.get_legend()
legend.set_title('critical temperature')
outliers = []
for i in range(len(molecules_latent_space)):
    if molecules_latent_space[i][0]< -50:
        outliers.append(i)

outliers
sc_dataframe['material'][outliers[1]]
from DataLoader import from_string_to_dict
error =[]
from_string_to_dict(sc_dataframe['material'][outliers[3]],error)
float(error[0][1]) > 150
regressor.rho.predict(atom_data.get_input(sc_dataframe['material'][outliers[1]]))
supercon[supercon['material'] == sc_dataframe['material'][outliers[0]]][]
sc_dataframe['material'][outliers[0]]
supercon = pd.read_csv('../../data/raw/SuperCon_database.dat',delimiter = r"\s+",names = ['material','critical_temp'])
#supercon = pd.read_csv('../../data/raw/supercon_material_temp.csv')
#supercon.rename(columns = {'material':'formula','critical_temp':'tc'},inplace=True)
#remove rows with nan value on tc
supercon = supercon.dropna()
#get duplicated row aggregating them with mean value on critical temperature
duplicated_row = supercon[supercon.duplicated(subset = ['formula'],keep = False)].groupby('formula').aggregate({'tc':'mean'}).reset_index()
#drop all the duplicated row
supercon.drop_duplicates(subset = ['formula'],inplace = True,keep = False)
#compose the entire dataset
supercon= supercon.append(duplicated_row,ignore_index=True)
