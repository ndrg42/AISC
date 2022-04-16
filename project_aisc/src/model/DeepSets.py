#
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
#
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.experimental.set_synchronous_execution(False)

from tensorflow import keras
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,Input,Add,LSTM,Conv1D,Flatten,Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError,Precision
import numpy as np
from sklearn.metrics import mean_squared_error as sk_mean_squared_error,r2_score as sk_r2_score
from sklearn.metrics import accuracy_score as sk_accuracy_score,precision_score as sk_precision_score
from sklearn.metrics import recall_score as sk_recall_score, f1_score as sk_f1_score
import sys
from mendeleev import element
import pandas as pd
sys.path.append('../data')
sys.path.append('../features')
import DataLoader
import Processing
from Processing import DataProcessor
import datetime
import kerastuner as kt
from kerastuner.tuners import Hyperband
import yaml
from yaml import Loader

#Implement class best Model
#Implement config file

with open('/home/claudio/AISC/project_aisc/config/model_config.yaml') as file:
    model_config = yaml.load(file,Loader)

# def build_phi(input_dim,latent_dim):
#     """Return phi model of rho(sum_i phi(atom_i))"""
#
#     input_atom = Input(shape = (input_dim,))
#     x = Dense(300,activation = "relu")(input_atom)
#     x = Dense(300,activation = "relu")(x)
#     output = Dense(latent_dim,activation = "linear",activity_regularizer = 'l1')(x)
#
#     return Model(inputs = input_atom,outputs = output)

def build_phi(input,layers,output):
    """Return phi model of rho(sum_i phi(atom_i))"""

    input_atom = Input(shape = (input,))
    x = input_atom

    for layer in layers:
        x = Dense(**layer)(x)

    output = Dense(**output)(x)

    return Model(inputs =input_atom,outputs = output)

def build_rho(input,layers,output):
    """Return rho model of rho(sum_i phi(atom_i))"""

    atom_representation = Input(shape = (input,))
    x = atom_representation

    for layer in layers:
        x = Dense(**layer)(x)

    output = Dense(**output)(x)

    return Model(inputs = atom_representation,outputs = output)

def get_linear_deepset_regressor(input_dim = 32,latent_dim=300,learning_rate=0.001):

    linear_regressor_deepset = DeepSetLinearModel(input_dim,latent_dim,mode = 'regression')
    linear_regressor_deepset.compile(optimizer= Adam(learning_rate = learning_rate),
                              loss= 'mean_squared_error',
                              metrics=['mean_absolute_error',RootMeanSquaredError()]
                              )
    return linear_regressor_deepset

def get_linear_deepset_classifier(input_dim = 32,latent_dim=300,learning_rate=0.001):

    classifier_deepset = DeepSetLinearModel(input_dim,latent_dim,mode = 'classification')
    classifier_deepset.compile(optimizer= Adam(learning_rate = learning_rate),
                              loss= 'binary_crossentropy',
                              metrics=['accuracy', Precision()]
                              )
    return classifier_deepset


class DeepSetModel(tf.keras.Model):
    """DeepSet model"""

    def __init__(self,phi_setup=model_config['phi setup'],rho_setup=model_config['regressor rho setup']):
        super(DeepSetModel,self).__init__()
        self.phi = build_phi(**phi_setup)
        self.rho = build_rho(**rho_setup)


    def call(self,atoms_input):

        phi_outputs = [self.phi(input) for input in atoms_input]
        material_representation = Add()(phi_outputs)
        rho_output = self.rho(material_representation)

        return rho_output

def get_deepset_regressor(phi_setup = model_config['phi setup'],
                          rho_setup = model_config['regressor rho setup'],
                          regressor_setup = model_config['regressor setup'],
                          ):

    regressor_deepset = DeepSetModel(phi_setup,rho_setup)
    regressor_deepset.compile(optimizer= regressor_setup['optimizer'](regressor_setup['learning rate']),
                              loss= regressor_setup['loss'],
                              metrics=[metric if isinstance(metric, str) else metric() for metric in regressor_setup['metrics']],
                              )

    return regressor_deepset


def get_deepset_classifier(input_dim = 33,latent_dim=300,learning_rate=0.001,model_setup=model_config['phi setup']):

    classifier_deepset = DeepSetModel(input_dim,latent_dim,mode='classification',model_setup = model_setup)
    classifier_deepset.compile(optimizer= Adam(learning_rate = learning_rate),
                              loss= 'binary_crossentropy',
                              metrics=['accuracy', Precision()]
                              )
    return classifier_deepset


class DeepSetLinearModel(DeepSetModel):

    def __init__(self,input_dim,latent_dim,mode='regression'):
        super(DeepSetLinearModel,self).__init__(input_dim,latent_dim,mode)

    def call(self,atoms_input):

        phi_outputs = [Multiply()([tf.expand_dims(input[:,-1],1),self.phi(input[:,:-1])]) for input in atoms_input]
        material_representation = Add()(phi_outputs)
        rho_output = self.rho(material_representation)

        return rho_output

def get_deepsetsecondorder_regressor(input_atom_dim = 33,
                                     input_interaction_dim=33,
                                     latent_dim=300,
                                     mode='regression',
                                     learning_rate=0.001):

    regressor_deepsetsecondorder = DeepSetSecondOrder(input_atom_dim=input_atom_dim,
                                           input_interaction_dim=input_interaction_dim,
                                           latent_dim=latent_dim,
                                           mode = mode)

    regressor_deepsetsecondorder.compile(optimizer= Adam(learning_rate = learning_rate),
                              loss= 'mean_squared_error',
                              metrics=['mean_absolute_error',RootMeanSquaredError()]
                              )

    return regressor_deepsetsecondorder


class DeepSetSecondOrder(DeepSetModel):

    def __init__(self,input_atom_dim,input_interaction_dim,latent_dim,mode,):
        super(DeepSetSecondOrder,self).__init__(input_atom_dim,latent_dim,mode='regression')
        self.interaction = DeepSetModel(input_interaction_dim,latent_dim,mode='regression')

    def call(self,atom_interaction_input):

        atoms_input = atom_interaction_input[:-1]
        interactions_input = atom_interaction_input[-1]

        phi_outputs = [self.phi(input) for input in atoms_input]
        linear_representation = Add()(phi_outputs)

        interaction_outputs = [self.interaction(input) for input in interactions_input]
        interactions_represetation = Add()(interaction_outputs)

        material_representation = Add()([linear_representation,interactions_represetation])
        rho_output = self.rho(material_representation)

        return rho_output


with open('/home/claudio/AISC/project_aisc/config/avaible_model_config.yaml') as file:
    #Load a dictionary contaning the model's name and the function to initialize them
    avaible_model = yaml.load(file,Loader)

def get_model(model='classifier'):
    """Retrive and return the specified model using avaible_model (dcit) as a switch controll."""

    model_builder = avaible_model.get(model)
    try:
        return model_builder()
    except:
        print('Model not found.')



# class VisualModel:
#
#     def __init__(self,history):
#         self.history = history
#
#     def show_metrics(self):
#
#         plt.plot(self.history.history['mean_absolute_error'])
#         plt.plot(self.history.history['val_mean_absolute_error'])
#         plt.title('Model mean_absolute_error')
#         plt.ylabel('mean_absolute_error')
#         plt.xlabel('Epoch')
#         plt.legend(['Train','Validation'],loc = 'upper left')
#         plt.show()
#
#     def show_loss(self):
#
#         plt.plot(self.history.history['loss'])
#         plt.plot(self.history.history['val_loss'])
#         plt.title('Model loss')
#         plt.ylabel('Loss')
#         plt.xlabel('Epoch')
#         plt.legend(['Train','Validation'],loc = 'upper left')
#         plt.show()
#

#
#
# class SubRegressorDeepSet2Order(tf.keras.Model):
#
#     def __init__(self):
#         super(SubRegressorDeepSet2Order,self).__init__()
#         self.phi = build_phi()
#         self.rho = build_rho()
#         self.phi_int = build_phi()
#         self.rho_int = self.build_rho_int()
#
#
#     def call(self,input_tensor):
#         single_inputs = input_tensor[:-1]
#         int_inputs = input_tensor[-1]
#         phi_outputs = [self.phi(input) for input in single_inputs]
#         y = Add()(phi_outputs)
#
#         rho_int_outputs = [self.rho_int(inputs) for inputs in int_inputs]
#         z = Add()(rho_int_outputs)
#         y = Add()([y,z])
#
#         return self.rho(y)
#
#
#     def build_rho_int(self):
#
#         inputs= [Input(33) for i in range(2)]
#         outputs = [self.phi_int(i) for i in inputs]
#
#         y = Add()(outputs)
#         y = Dense(300,activation = "relu")(y)
#         y = Dense(300,activation = "relu")(y)
#         output = Dense(300,activation = "linear",activity_regularizer = 'l1')(y)
#
#         return Model(inputs = inputs,outputs = output)
#

class VisualDeepSet():

    def __init__(self):
        self.history = None

    def show_metrics(self):

        plt.plot(self.history.history['mean_absolute_error'])
        plt.plot(self.history.history['val_mean_absolute_error'])
        plt.title('Model mean_absolute_error')
        plt.ylabel('mean_absolute_error')
        plt.xlabel('Epoch')
        plt.legend(['Train','Validation'],loc = 'upper left')
        plt.show()

    def show_loss(self):

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train','Validation'],loc = 'upper left')
        plt.show()




class BaseDeepSet(VisualDeepSet):

    def __init__(self,supercon_data_processed,latent_dimension=300):
        self.number_inputs = len(supercon_data_processed)
        self.input_dimension = supercon_data_processed[0].shape[1]
        self.latent_dimension = latent_dimension
        self.phi = None
        self.rho = None
        self.history = None


    def build_phi(self):

        input_atom = Input(shape = (self.input_dimension))
        x = Dense(300,activation = "relu")(input_atom)
        x = Dense(300,activation = "relu")(x)
        x = Dense(self.latent_dimension,activation = "linear",activity_regularizer = 'l1')(x)

        self.phi = Model(inputs = input_atom,outputs = x)


    def build_rho(self):
        pass

    def predict(self,X):

        return self.rho.predict(X)


    def fit_model(self,X,y,X_val,y_val,callbacks = [],epochs= 50,batch_size = 64,patience=5):

        #callbacks.append(EarlyStopping(monitor = 'val_loss',min_delta = 0.003,patience = patience, restore_best_weights = True))
        self.history = self.rho.fit(x = X,y = y,epochs =epochs, batch_size = batch_size,shuffle = True ,validation_data =(X_val,y_val),callbacks = callbacks )



class RegressorDeepSet(BaseDeepSet):

    def __init__(self,supercon_data_processed,latent_dimension=300):
        super().__init__(supercon_data_processed,latent_dimension)
        self.rho = None

    def build_rho(self):

        inputs= [Input(self.input_dimension) for i in range(self.number_inputs)]
        outputs = [self.phi(i) for i in inputs]

        y = Add()(outputs)
        y = Dense(300,activation = "relu")(y)
        y = Dense(300,activation = "relu")(y)
        y = Dense(300,activation = "relu")(y)
        y = Dense(300,activation = "relu")(y)
        y = Dense(100,activation = "relu")(y)
        output = Dense(1,activation = "relu",activity_regularizer = 'l1')(y)

        self.rho = Model(inputs = inputs,outputs = output)


    def build_model(self,learning_rate=0.001):

        self.build_phi()
        self.build_rho()
        self.rho.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate),
                          loss='mse',
                          metrics=['mean_absolute_error',tf.keras.metrics.RootMeanSquaredError()])



    def evaluate_model(self,X_test,y_test):

        y_pred = self.predict(X_test)
        print(f"Mean Squared Error: {sk_mean_squared_error(y_test,y_pred)}\n"
              f"Root Mean Squared Error: {sk_mean_squared_error(y_test,y_pred,squared=False)}\n"
              f"R^2: {sk_r2_score(y_test,y_pred)}")

    def naive_classification(self,X_test,threshold=0):

        y_pred = self.predict(X_test)
        y_pred = tf.where(y_pred>threshold,1,0)

        return y_pred

    def naive_classification_metrics(self,threshold,X_test,y_test):

        y_pred = self.predict(X_test)
        y_pred = tf.where(y_pred>threshold,1,0)
        print(f"Accuracy: {sk_accuracy_score(y_test,y_pred)} \nPrecision: {sk_precision_score(y_test,y_pred)}"
              f"Recall: {sk_recall_score(y_test,y_pred)} \nF1: {sk_f1_score(y_test,y_pred)}")




import itertools

class RegressorDeepSet2Order(BaseDeepSet):

    def __init__(self,supercon_data_processed,latent_dimension):
        super().__init__(supercon_data_processed,latent_dimension)
        self.rho = None
        self.phi_int = None
        self.rho_int = None

    def build_rho_int(self):

        inputs= [Input(self.input_dimension) for i in range(2)]
        outputs = [self.phi_int(i) for i in inputs]

        y = Add()(outputs)
        y = Dense(300,activation = "relu")(y)
        y = Dense(300,activation = "relu")(y)
        output = Dense(300,activation = "linear",activity_regularizer = 'l1')(y)

        self.rho_int = Model(inputs = inputs,outputs = output)



    def build_rho(self):

        inputs= [Input(self.input_dimension) for i in range(self.number_inputs)]
        outputs = [self.phi(i) for i in inputs]

        outputs_int = [self.rho_int(interaction) for interaction in itertools.combinations(inputs,2)]

        z = Add()(outputs_int)
        y = Add()(outputs)
        y = y+z
        y = Dense(300,activation = "relu")(y)
        y = Dense(300,activation = "relu")(y)
        y = Dense(300,activation = "relu")(y)
        y = Dense(300,activation = "relu")(y)
        y = Dense(100,activation = "relu")(y)
        output = Dense(1,activation = "relu",activity_regularizer = 'l1')(y)

        self.rho = Model(inputs = inputs,outputs = output)


    def build_model(self,learning_rate=0.001):

        self.build_phi()
        self.phi_int = self.phi
        self.build_phi()
        self.build_rho_int()
        self.build_rho()
        self.rho.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate),
                          loss='mse',
                          metrics=['mae',tf.keras.metrics.RootMeanSquaredError()])



    def evaluate_model(self,X_test,y_test):


        y_pred = self.predict(X_test)

        print(f"Mean Squared Error: {sk_mean_squared_error(y_test,y_pred)}\n"
              f"Root Mean Squared Error: {sk_mean_squared_error(y_test,y_pred,squared=False)}\n"
              f"R^2: {sk_r2_score(y_test,y_pred)}")





class ClassifierDeepSet(BaseDeepSet):

    def __init__(self,supercon_data_processed,latent_dimension):
        super().__init__(supercon_data_processed,latent_dimension)
        self.rho = None

    def build_rho(self):

        inputs= [Input(self.input_dimension) for i in range(self.number_inputs)]
        outputs = [self.phi(i) for i in inputs]

        y = Add()(outputs)
        y = Dense(300,activation = "relu")(y)
        y = Dense(300,activation = "relu")(y)
        y = Dense(300,activation = "relu")(y)
        y = Dense(300,activation = "relu")(y)
        y = Dense(100,activation = "relu")(y)
        output = Dense(1,activation = "sigmoid",activity_regularizer = 'l1')(y)

        self.rho = Model(inputs = inputs,outputs = output)

    def build_model(self,learning_rate=0.0001):

        self.build_phi()
        self.build_rho()
        self.rho.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate),
                          loss='binary_crossentropy',
                          metrics=['accuracy',tf.keras.metrics.Precision()])


    def evaluate_classificator(self,X_test,y_test):

        print("sparse_categorical_crossentropy: ",self.rho_classificator.evaluate(X_test,y_test,verbose=0)[0],"\nAccuracy: ",self.rho_classificator.evaluate(X_test,y_test,verbose=0)[1])
        print("Accuracy on test: ",np.sqrt(self.rho_classificator.evaluate(X_test,y_test,verbose = 0)[1]))


class DeepSet():

    def __init__(self,DataProcessor,latent_dim,freeze_latent_dim_on_tuner = False):


        self.latent_dim = latent_dim
        self.n_inputs = len(DataProcessor.dataset)
        self.input_dim = DataProcessor.dataset[0].shape[1]
        self.phi = None
        self.phi_classificator = None
        self.rho_classificator = None
        self.rho = None
        self.history = None
        self.history_classification = None
        self.freeze_latent_dim_on_tuner = freeze_latent_dim_on_tuner


    def build_phi(self):

        input_atom = Input(shape = (self.input_dim))
        n_element = input_atom[:,-1:]
        c = input_atom[:,:-1]
        x = Dense(300,activation = "relu")(c)
        #x = Dropout(0.5)(x)
        #x = BatchNormalization()(x)
        x = Dense(300,activation = "relu")(x)
        #x = Dropout(0.3)(x)
        x = Dense(300,activation = "relu")(x)
        #x = Dropout(0.3)(x)
        y = Dense(self.latent_dim,activation = "linear",activity_regularizer = 'l1')(x)
        y = tf.math.multiply(y,n_element)
        #y = tf.reshape(y,(tf.shape(input_atom)[0],self.latent_dim))
        #y = tf.keras.layers.Lambda(lambda x: x*n_element)(y)
        self.phi = Model(inputs = input_atom,outputs = y)



    def build_rho_regressor(self):

        inputs= [Input(self.input_dim) for i in range(self.n_inputs)]
        outputs = [self.phi(i) for i in inputs]

        y = Add()(outputs)
        y = Dense(300,activation = "relu")(y)
        #y = BatchNormalization()(y)
        #y = Dropout(0.5)(y)
        y = Dense(300,activation = "relu")(y)
        #y = Dropout(0.5)(y)
        y = Dense(300,activation = "relu")(y)
        y = Dense(300,activation = "relu")(y)
        #y = BatchNormalization()(y)
        y = Dense(100,activation = "relu")(y)
        output = Dense(1,activation = "relu",activity_regularizer = 'l1')(y)

        self.rho = Model(inputs = inputs,outputs = output)

    def build_regressor(self,learning_rate=0.0001):

        self.build_phi()
        self.build_rho_regressor()
        self.rho.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate),
                          loss='mse',
                          metrics=['mae',tf.keras.metrics.RootMeanSquaredError()])



    def build_rho_classifier(self):

        inputs= [Input(self.input_dim) for i in range(self.n_inputs)]
        outputs = [self.phi(i) for i in inputs]

        y = Add()(outputs)
        y = Dense(300,activation = "relu")(y)
        #y = BatchNormalization()(y)
        #y = Dropout(0.5)(y)
        y = Dense(300,activation = "relu")(y)
        #y = Dropout(0.5)(y)
        y = Dense(300,activation = "relu")(y)
        #y = BatchNormalization()(y)
        y = Dense(100,activation = "relu")(y)
        output = Dense(1,activation = "sigmoid",activity_regularizer = 'l1')(y)

        self.rho = Model(inputs = inputs,outputs = output)

    def build_classifier(self,learning_rate=0.0001):

        self.build_phi()
        self.build_rho_classifier()
        self.rho.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate),
                          loss='binary_crossentropy',
                          metrics=['accuracy',tf.keras.metrics.Precision()])



    def fit_model(self,X,y,X_val,y_val,callbacks,epochs= 50,batch_size = 64,patience=5):

        #early_stopping_callback = EarlyStopping(monitor = 'val_mean_absolute_error',min_delta = 0.05,patience = 5, restore_best_weights = True)
        callbacks.append(EarlyStopping(monitor = 'val_loss',min_delta = 0.003,patience = patience, restore_best_weights = True))
        self.history = self.rho.fit(x = X,y = y,epochs =epochs, batch_size = batch_size,shuffle = True ,validation_data =(X_val,y_val),callbacks = callbacks )

    def evaluate_model(self,X_test,y_test):

        R2=1-np.square((y_test-np.reshape(self.rho.predict(X_test),y_test.shape))).sum()/np.square((y_test - y_test.mean())).sum()

        print("MSE: ",self.rho.evaluate(X_test,y_test,verbose=0)[0],"\nMAE: ",self.rho.evaluate(X_test,y_test,verbose=0)[1])
        print("RMSE: ",np.sqrt(self.rho.evaluate(X_test,y_test,verbose = 0)[0]),"\nR^2:",R2)

    def save_model(self,path,name):
        self.rho.save(path+name)

    def load_model(self,path,name):

        from tensorflow.keras.models import load_model as tf_load_model
        self.rho = tf_load_model(path+name)




    def model_builder(self,hp):

        model = self.rho_builder(hp)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-4,1e-5,1e-6])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = hp_learning_rate),
                          loss='mse',
                          metrics=[tf.keras.metrics.RootMeanSquaredError(),'mae'])

        return model

    def phi_builder(self,hp):

        input_atom = Input(shape = (self.input_dim))
        #n_element = input_atom[:,-1:]
        #c = input_atom[:,:-1]
        x = Dense(hp.Int('units_-1',min_value = 32,max_value = 1024,step = 32),activation='relu')(input_atom)
        layers = hp.Int('layers_1',min_value = 5,max_value = 25,step = 1)
        dropout_batch = hp.Choice('dropout_batch_phi',[True,False])
        for layer in range(layers):
            x = Dense(hp.Int('units_'+str(layer),min_value = 32,max_value = 1024,step = 32),activation='relu')(x)
            if dropout_batch:
                x = BatchNormalization()(x)
                x = Dropout(hp.Float('dropout',0,0.4,step = 0.2))(x)

        #if self.freeze_latent_dim_on_tuner:
        #    x = Dense(self.latent_dim,activation='linear')(x)
        x = Dense(hp.Int('units_'+str(layer+1),min_value = 32,max_value = 1024,step = 32),activation='relu')(x)
        #x = tf.math.multiply(x,n_element)


        phi = Model(inputs = input_atom,outputs = x)
        layers = layers +1
        return phi,layers


    def rho_builder(self,hp):

        phi,n_layers_phi = self.phi_builder(hp)
        inputs= [Input(self.input_dim) for i in range(self.n_inputs)]
        outputs = [phi(i) for i in inputs]

        y = Add()(outputs)
        dropout_batch = hp.Choice('dropout_batch_rho',[True,False])
        layers = hp.Int('layers_2',min_value = 8,max_value = 23,step = 1)
        for layer in range(layers):
            y = Dense(hp.Int('units_'+str(n_layers_phi+layer),min_value = 32,max_value = 1024,step = 32),activation='relu')(y)
            if dropout_batch:
                y = BatchNormalization()(y)
                y = Dropout(hp.Float('dropout',0,0.4,step = 0.2))(y)

        #activity_regularizer = hp.Choice('regularizer_rho',['l1','l2'])
        #output = Dense(1,activation = "sigmoid",activity_regularizer = activity_regularizer)(y)
        y = Dense(hp.Int('units_'+str(n_layers_phi+layer+1),min_value = 32,max_value = 512,step = 32),activation='relu')(y)
        output = Dense(1,activation = "linear",activity_regularizer = 'l2')(y)
        rho = Model(inputs = inputs,outputs = output)

        return rho

#    def get_best_model(self,X,Y,X_val,Y_val,max_epochs=400,epochs=400,num_best_model=1):

    def get_best_model(self,X,Y,max_epochs=400,epochs=400,num_best_model=1):

        import datetime
        import os

        date = datetime.datetime.now()

        directory = '../../models/best_model_'+date.strftime("%d")+"-"+date.strftime("%m")

        DIR = '../../models/best_model_'+date.strftime("%d")+"-"+date.strftime("%m")

        try:
            n_best_model_per_day =len([name for name in os.listdir(DIR)])
        except:
            os.makedirs(DIR)
            n_best_model_per_day = 0

        project_name = 'model_regressor'+date.strftime("%d")+"-"+date.strftime("%m")+"-"+str(n_best_model_per_day)

        tuner = kt.Hyperband(_model_builder,
                     objective='val_loss',
                     max_epochs=max_epochs,
                     directory=directory,
                     project_name=project_name
                     )

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

        tuner.search(X,Y, epochs=epochs, validation_split=0.2, callbacks=[stop_early])
    #    tuner.search(X,Y, epochs=epochs, validation_data=(X_val,Y_val), callbacks=[stop_early])

        model = tuner.get_best_models(num_models=num_best_model)

        self.rho = model[0]

        return model

    def load_best_model(self,directory,project_name,max_epochs=400,more_models = False,num_models = 1):

        tuner = kt.Hyperband(self.model_builder,
                     objective='val_loss',
                     max_epochs=max_epochs,
                     directory=directory,
                     project_name=project_name
                     )

        tuner.reload()

        if more_models:
            return tuner.get_best_models(num_models=num_models)
            del tuner
        else:
            self.rho = tuner.get_best_models(num_models=1)[0]
            del tuner

    def load_best_architecture(self,directory,project_name,max_epochs=400):

        tuner = kt.Hyperband(self.model_builder,
                     objective='val_loss',
                     max_epochs=max_epochs,
                     directory=directory,
                     project_name=project_name
                     )

        tuner.reload()

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.rho = tuner.hypermodel.build(best_hps)

        del tuner



    def naive_classificator(self,threshold,X_test,y_test):

        limit = threshold

        #true_positive: valore maggiore del threshold
        true_positive = 0
        #true_negative: valore minore del threshold
        true_negative = 0
        false_positive = 0
        false_negative = 0

        val = self.rho.predict(X_test)

        for i in range(X_test[0].shape[0]):

            if y_test[i] > limit:

                if val[i] > limit:
                    true_positive +=1
                if val[i] <= limit:
                    false_negative +=1

            if y_test[i] <= limit:

                if val[i] > limit:
                    false_positive +=1

                if val[i] <= limit:
                    true_negative +=1

        Accuracy = (true_positive+true_negative)/(true_positive+true_negative+false_negative+false_positive)
        Precision = (true_positive)/(true_positive+false_positive)
        Recall = (true_positive)/(true_positive+false_negative)
        F1 = 2*(Recall*Precision)/(Recall + Precision)

        print("Accuracy: {} \nPrecision: {} \nRecall: {} \nF1:{}".format(Accuracy,Precision,Recall,F1))

        return true_positive,true_negative,false_positive,false_negative


    def confusion_matrix(self,X_test,Y_test):

        data = {}
        data['observed'] = list(Y_test>10)
        data['predicted'] = list(np.reshape((self.rho.predict(X_test) > 10),(Y_test.shape[0])))

        data = pd.DataFrame(data).replace(False,0)


        import sklearn
        cm = sklearn.metrics.confusion_matrix(data['observed'],data['predicted'])
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                       display_labels=['non sc','sc'])

        disp.plot()





    def visual_model_perform(self):

        plt.plot(self.history.history['mean_absolute_error'])
        plt.plot(self.history.history['val_mean_absolute_error'])
        plt.title('Model mean_absolute_error')
        plt.ylabel('mean_absolute_error')
        plt.xlabel('Epoch')
        plt.legend(['Train','Validation'],loc = 'upper left')
        plt.show()

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train','Validation'],loc = 'upper left')
        plt.show()


    def input_from_dict(self,d_m):

        input_dim = self.input_dim

        try:
            if self.Atom == None:
                super().build_Atom()
        except:
            pass

        nulla = np.zeros(input_dim)
        entrata = []
        for j in range(self.max_lunghezza):
            if j < len(d_m):
                entrata.append(np.array(self.Atom.loc[(element(list(d_m.keys())[j]).atomic_number -1)].append(pd.Series(list(d_m.values())[j]))))
            else:
                entrata.append(np.array(pd.Series(nulla)))

        e = np.array(entrata)
        e = list(np.expand_dims(e,axis = 1))

        return e


    def get_dict_material(self,compound):

        num = ['0','1','2','3','4','5','6','7','8','9','.']
        key=''
        d= {}
        value = ''
        ok = False
        compound = compound
        for s in compound:
            if s not in num:
                if ok == True:
                    d.update({key:float(value)})
                    key = ''
                    value = ''
                    ok = False

                key = key+s
            if s in num:
                value = value +s
                ok = True
        if ok == False:
            d.update({key:1.0})
        if ok == True:
            d.update({key:float(value)})
        return d


    def build_phi_classificator(self):

        input_atom = Input(shape = (self.input_dim))
        x = BatchNormalization()(input_atom)
        x = Dropout(0.5)(x)
        x = Dense(300,activation = "relu")(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Dense(300,activation = "relu")(x)
        x = Dropout(0.3)(x)
        x = Dense(300,activation = "relu")(x)
        y = Dense(self.latent_dim,activation = "linear",activity_regularizer = 'l1')(x)

        self.phi_classificator = Model(inputs = input_atom,outputs = y)


    def build_rho_classificator(self):

        inputs= [Input(self.input_dim) for i in range(self.n_inputs)]
        outputs = [self.phi_classificator(i) for i in inputs]

        y = Add()(outputs)
        y = Dense(300,activation = "relu")(y)
        y = BatchNormalization()(y)
        y = Dropout(0.5)(y)
        y = Dense(300,activation = "relu")(y)
        y = Dropout(0.5)(y)
        y = Dense(300,activation = "relu")(y)
        y = BatchNormalization()(y)
        y = Dense(100,activation = "relu")(y)
        output = Dense(2,activation = "sigmoid",activity_regularizer = 'l2')(y)
        self.rho_classificator = Model(inputs = inputs,outputs = output)

    def build_classificator(self):

        self.build_phi_classificator()
        self.build_rho_classificator()

        self.rho_classificator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001),
                                    loss='sparse_categorical_crossentropy',
                                    metrics=['accuracy'])


    def fit_clafficator(self,X,y,X_val,y_val):

        early_stopping_callback = EarlyStopping(monitor = 'accuracy',min_delta = 0.05,patience = 5, restore_best_weights = True)
        self.history_classification = self.rho_classificator.fit(x = X,y = y,epochs =50, batch_size = 64,shuffle = True ,validation_data =(X_val,y_val),callbacks = [early_stopping_callback] )

    def evaluate_classificator(self,X_test,y_test):

        print("sparse_categorical_crossentropy: ",self.rho_classificator.evaluate(X_test,y_test,verbose=0)[0],"\nAccuracy: ",self.rho_classificator.evaluate(X_test,y_test,verbose=0)[1])
        print("Accuracy on test: ",np.sqrt(self.rho_classificator.evaluate(X_test,y_test,verbose = 0)[1]))



    def build_hybrid_phi(self):

        input_atom = Input(shape = (self.input_dim,1))
        #x = BatchNormalization()(input_atom)
        #x = Dropout(0.5)(x)
        x = Conv1D(32,3,padding='same',activation = "relu")(input_atom)
        #x = BatchNormalization()(x)
        #x = Dropout(0.3)(x)

        x = Conv1D(64,3,padding='same',activation = "relu")(x)
        #x = BatchNormalization()(x)
        #x = Dropout(0.3)(x)


        x = Conv1D(64,3,padding='same',activation = "relu")(x)
        #x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)

        #x = LSTM(128,return_sequences=True)(x)
        x = LSTM(64,return_sequences=False)(x)

        x = Dense(300,activation = "relu")(x)

        y = Dense(self.latent_dim,activation = "linear",activity_regularizer = 'l1')(x)

        model = Model(inputs = input_atom,outputs = y)

        return model



    def build_hybrid_rho(self):

        inputs= [Input(self.input_dim,1) for i in range(self.n_inputs)]
        phi = self.build_hybrid_phi()
        outputs = [phi(i) for i in inputs]

        y = Add()(outputs)
        y = tf.expand_dims(y,axis=2)
        y = Conv1D(32,3,padding='same',activation = "relu")(y)
        #y = BatchNormalization()(y)
        #y = Dropout(0.3)(y)

        y = Conv1D(64,3,padding='same',activation = "relu")(y)
        #y = BatchNormalization()(y)
        #y = Dropout(0.3)(y)


        y = Conv1D(64,3,padding='same',activation = "relu")(y)
        #y = BatchNormalization()(y)
        #y = Dropout(0.5)(y)

        #y = LSTM(128,return_sequences=True)(y)
        y = LSTM(64,return_sequences=False)(y)


        y = Dense(300,activation = "relu")(y)

        y = Dense(self.latent_dim,activation = "linear",activity_regularizer = 'l2')(y)

        self.rho = Model(inputs = inputs,outputs = y)

    def build_hybrid_model(self,learning_rate=0.0001):

        self.build_hybrid_phi()
        self.build_hybrid_rho()
        self.rho.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate),
                          loss='mean_squared_error',
                          metrics=['mean_absolute_error'])




def _model_builder(hp):

    model = _DeepSetSecondOrder(hp)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4,1e-5,1e-6])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = hp_learning_rate),
                      loss='mse',
                      metrics=[tf.keras.metrics.RootMeanSquaredError(),'mae'])

    return model


def _build_phi(hp,input_atom_dim=33):

    input_atom = Input(shape = (33))
    x = Dense(hp.Int('units_-1',min_value = 32,max_value = 512,step = 32),activation='relu')(input_atom)
    layers = hp.Int('layers_1',min_value = 3,max_value = 25,step = 1)
    dropout_batch = hp.Choice('dropout_batch_phi',[True,False])
    for layer in range(layers):
        x = Dense(hp.Int('units_'+str(layer),min_value = 32,max_value = 512,step = 32),activation='relu')(x)
        if dropout_batch:
            x = BatchNormalization()(x)
            x = Dropout(hp.Float('dropout',0,0.4,step = 0.2))(x)

    x = Dense(300,activation='linear')(x)

    phi = Model(inputs = input_atom,outputs = x)
    layers = layers +1
    return phi

def _build_rho(hp):

    atom_representation = Input(shape = (300,))
    y = Dense(hp.Int('units_-10',min_value = 32,max_value = 512,step = 32),activation='relu')(atom_representation)
    dropout_batch = hp.Choice('dropout_batch_rho',[True,False])
    layers = hp.Int('layers_2',min_value = 8,max_value = 23,step = 1)
    for layer in range(layers):
        y = Dense(hp.Int('units0_'+str(+layer),min_value = 32,max_value = 1024,step = 32),activation='relu')(y)
        if dropout_batch:
            y = BatchNormalization()(y)
            y = Dropout(hp.Float('dropout',0,0.4,step = 0.2))(y)

    #activity_regularizer = hp.Choice('regularizer_rho',['l1','l2'])
    #output = Dense(1,activation = "sigmoid",activity_regularizer = activity_regularizer)(y)
    y = Dense(hp.Int('units0_'+str(+layer+1),min_value = 32,max_value = 512,step = 32),activation='relu')(y)
    output = Dense(1,activation = "relu",activity_regularizer = 'l2')(y)
    rho = Model(inputs = atom_representation,outputs = output)

    return rho

def _build_phi_int(hp,input_atom_dim=33):

    input_atom = Input(shape = (33))
    x = Dense(hp.Int('units_-11',min_value = 32,max_value = 512,step = 32),activation='relu')(input_atom)
    layers = hp.Int('layers_1',min_value = 1,max_value = 10,step = 1)
    dropout_batch = hp.Choice('dropout_batch_phi',[True,False])
    for layer in range(layers):
        x = Dense(hp.Int('units_1'+str(layer),min_value = 32,max_value = 512,step = 32),activation='relu')(x)
        if dropout_batch:
            x = BatchNormalization()(x)
            x = Dropout(hp.Float('dropout',0,0.4,step = 0.2))(x)

    x = Dense(300,activation='linear')(x)

    phi = Model(inputs = input_atom,outputs = x)
    layers = layers +1
    return phi

def _build_rho_int(hp):

    atom_representation = Input(shape = (300,))
    y = Dense(hp.Int('units_-10',min_value = 32,max_value = 512,step = 32),activation='relu')(atom_representation)
    dropout_batch = hp.Choice('dropout_batch_rho',[True,False])
    layers = hp.Int('layers_20',min_value = 1,max_value = 10,step = 1)
    for layer in range(layers):
        y = Dense(hp.Int('units1_1'+str(layer),min_value = 32,max_value = 1024,step = 32),activation='relu')(y)
        if dropout_batch:
            y = BatchNormalization()(y)
            y = Dropout(hp.Float('dropout',0,0.4,step = 0.2))(y)

    #activity_regularizer = hp.Choice('regularizer_rho',['l1','l2'])
    #output = Dense(1,activation = "sigmoid",activity_regularizer = activity_regularizer)(y)
    y = Dense(hp.Int('units1_1'+str(layer+1),min_value = 32,max_value = 512,step = 32),activation='relu')(y)
    output = Dense(1,activation = "linear",activity_regularizer = 'l2')(y)
    rho = Model(inputs = atom_representation,outputs = output)

    return rho



class _DeepSetSecondOrder(tf.keras.Model):

    def __init__(self,hp):
        super(_DeepSetSecondOrder,self).__init__()
        self.phi = _build_phi(hp)
        self.rho = _build_rho(hp)
        self.phi_int = _build_phi_int(hp)
        self.rho_int = _build_rho_int(hp)

    def call(self,atom_interaction_input):

        atoms_input = atom_interaction_input[:-1]
        interactions_input = atom_interaction_input[-1]

        phi_outputs = [self.phi(input) for input in atoms_input]
        linear_representation = Add()(phi_outputs)

        interaction_outputs = [self.rho_int(Add()([self.phi_int(input[0]),self.phi_int(input[1])])) for input in interactions_input]
        interactions_represetation = Add()(interaction_outputs)

        material_representation = Add()([linear_representation,interactions_represetation])
        rho_output = self.rho(material_representation)

        return rho_output
