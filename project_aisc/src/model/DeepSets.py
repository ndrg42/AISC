from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,Input,Add,LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow import keras
import numpy as np
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


class DeepSet():

    def __init__(self,DataProcessor,latent_dim):


        self.latent_dim = latent_dim
        self.n_inputs = len(DataProcessor.dataset)
        self.input_dim = DataProcessor.dataset[0].shape[1]
        self.phi = None
        self.phi_classificator = None
        self.rho_classificator = None
        self.rho = None
        self.history = None
        self.history_classification = None


    def build_phi(self):

        input_atom = Input(shape = (self.input_dim))
        x = BatchNormalization()(input_atom)
        x = Dropout(0.5)(x)
        x = Dense(200,activation = "relu")(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Dense(200,activation = "relu")(x)
        x = Dropout(0.3)(x)
        x = Dense(200,activation = "relu")(x)
        y = Dense(self.latent_dim,activation = "linear",activity_regularizer = 'l1')(x)

        self.phi = Model(inputs = input_atom,outputs = y)


    def build_rho(self):

        inputs= [Input(self.input_dim) for i in range(self.n_inputs)]
        outputs = [self.phi(i) for i in inputs]

        y = Add()(outputs)
        y = Dense(300,activation = "relu")(y)
        y = BatchNormalization()(y)
        y = Dropout(0.5)(y)
        y = Dense(300,activation = "relu")(y)
        y = Dropout(0.5)(y)
        y = Dense(200,activation = "relu")(y)
        y = BatchNormalization()(y)
        y = Dense(100,activation = "relu")(y)
        output = Dense(1,activation = "linear",activity_regularizer = 'l1')(y)

        self.rho = Model(inputs = inputs,outputs = output)

    def build_model(self,learning_rate=0.0001):

        self.build_phi()
        self.build_rho()
        self.rho.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate),
                          loss='mean_squared_error',
                          metrics=['mean_absolute_error'])


    def fit_model(self,X,y,X_val,y_val,callbacks,epochs= 50,batch_size = 64,patience=5):

        #early_stopping_callback = EarlyStopping(monitor = 'val_mean_absolute_error',min_delta = 0.05,patience = 5, restore_best_weights = True)
        callbacks.append(EarlyStopping(monitor = 'val_mean_absolute_error',min_delta = 0.05,patience = patience, restore_best_weights = True))
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
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-4,1e-5])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = hp_learning_rate),
                          loss='mean_squared_error',
                          metrics=['mean_absolute_error'])

        return model

    def phi_builder(self,hp):
        input_atom = Input(shape = (self.input_dim))
        x = Dense(self.input_dim,kernel_initializer=tf.keras.initializers.Identity(),use_bias=False,activation='linear')(input_atom)
        layers = hp.Int('layers_1',min_value = 1,max_value = 5,step = 1)
        for layer in range(layers):
            x = Dense(hp.Int('units_'+str(layer),min_value = 32,max_value = 400,step = 32),activation='relu')(x)
        phi = Model(inputs = input_atom,outputs = x)

        return phi,layers

    def rho_builder(self,hp):

        phi,n_layers_phi = self.phi_builder(hp)
        inputs= [Input(self.input_dim) for i in range(self.n_inputs)]
        outputs = [phi(i) for i in inputs]

        y = Add()(outputs)
        layers = hp.Int('layers_2',min_value = 1,max_value = 5,step = 1)
        for layer in range(layers):
            y = Dense(hp.Int('units_'+str(n_layers_phi+layer),min_value = 32,max_value = 400,step = 32),activation='relu')(y)


        output = Dense(1,activation = "linear",activity_regularizer = 'l1')(y)
        rho = Model(inputs = inputs,outputs = output)

        return rho

    def get_best_model(self,X,Y,X_val,Y_val,max_epochs=5,epochs=5,num_best_model=1):

        import datetime
        import os

        date = datetime.datetime.now()

        directory = '../../models/best_model_'+date.strftime("%d")+"-"+date.strftime("%m")

        DIR = '../../models/best_model_'+date.strftime("%d")+"-"+date.strftime("%m")
        n_best_model_per_day =len([name for name in os.listdir(DIR)])
        project_name = 'model_'+date.strftime("%d")+"-"+date.strftime("%m")+"-"+str(n_best_model_per_day)

        tuner = kt.Hyperband(self.model_builder,
                     objective='val_loss',
                     max_epochs=max_epochs,
                     directory=directory,
                     project_name=project_name
                     )

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        tuner.search(X,Y, epochs=epochs, validation_data=(X_val,Y_val), callbacks=[stop_early])

        model = tuner.get_best_models(num_models=num_best_model)

        self.rho = model[0]

        return model

    def load_best_model(self,directory,project_name,max_epochs=5,more_models = False,num_models = 1):

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

    def load_best_architecture(self,directory,project_name,max_epochs=5):

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
        good=0
        bad=0
        j=0
        X_test_pred = []
        for i in range(X_test[0].shape[0]):
            X_test_pred = []
            for k in range(0,len(X_test)):
                d = X_test[k][i][:]
                d = np.expand_dims(d,axis = 1)
                d=np.moveaxis(d,0,1)
                X_test_pred.append(d)


            val = self.rho.predict(X_test_pred)
            if y_test[i] > limit:
                if val > limit:
                    good +=1
                if val < limit:
                    bad +=1

            if y_test[i] < limit:
                if val < limit:
                    good +=1
                if val > limit:
                    bad +=1

        print(good/X_test[0].shape[0])


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


    def get_input(self,compound):

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

        input_dim = self.input_dim
        nulla = np.zeros(input_dim)
        entrata = []

        try:
            if self.Atom == None:
                super().build_Atom()
        except:
            pass

        for j in range(self.max_lunghezza):
            if j < len(d):
                entrata.append(np.array(self.Atom.loc[(element(list(d.keys())[j]).atomic_number -1)].append(pd.Series(list(d.values())[j]))))
            else:
                entrata.append(np.array(pd.Series(nulla)))

        e = np.array(entrata)
        e = list(np.expand_dims(e,axis = 1))

        return e

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
