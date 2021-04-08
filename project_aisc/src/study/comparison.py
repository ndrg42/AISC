import sys
sys.path.append('/home/claudio/aisc/project_aisc/src/data')
sys.path.append('/home/claudio/aisc/project_aisc/src/features')
sys.path.append('/home/claudio/aisc/project_aisc/src/model')
import DataLoader
import Processing
from Processing import DataProcessor
import DeepSets
from DeepSets import DeepSet
import numpy as np
import pandas as pd
import csv
#%%
path_comparison = 'src/study/comparison_data'

ptable = DataLoader.PeriodicTable()
sc_dataframe = DataLoader.SuperCon(sc_path = 'data/raw/unique_m.csv')

atom_data = Processing.DataProcessor(ptable, sc_dataframe)
path = 'data/processed/'
atom_data.load_data_processed(path + 'dataset_elaborated.csv')
atom_data.load_data_processed(path + 'dataset_label_elaborated.csv')

dimensions = [2,10]
n_cycles = 2
import csv
for latent_dim in dimensions:
    for count in range(n_cycles):

        X,X_val,y,y_val = atom_data.train_test_split(atom_data.dataset,np.array(atom_data.t_c),test_size = 0.2)
        X,X_test,y,y_test = atom_data.train_test_split(X,y,test_size = 0.2)

        model = DeepSet(DataProcessor = atom_data,latent_dim = latent_dim)
        model.build_model()

        callbacks = []
        model.fit_model(X,y,X_val,y_val,callbacks= callbacks)

        with open(path_comparison+'score_diff_dim.csv', mode='a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            csv_writer.writerow([model.rho.evaluate(X_test,y_test,verbose=0)[1], model.rho.evaluate(X_test,y_test,verbose = 0)[0], np.sqrt(model.rho.evaluate(X_test,y_test,verbose = 0)[0]),latent_dim])
#%%

score_complete = pd.read_csv(path_comparison+'score_diff_dim.csv',header = None)

punteggi = {}
observable = {'MAE':0,'MSE':1,'RMSE':2}
chosen_observable = 'MAE'
score_observed = observable[chosen_observable]
dim = 0
for index in range(3):
    punteggi[score_complete.iloc[dim][3]] = chosen_observable + ' medio: {}; std: {}'.format(score_complete[index:index+n_cycles][score_observed].mean(),score_complete[index:index+n_cycles][score_observed].std())
    dim = dim+n_cycles

punteggi
dim
for index in range(3):
    punteggi[score_complete.iloc[dim][3]] = chosen_observable + ' medio: {}; std: {}'.format(score_complete[index:index+n_cycles][score_observed].mean(),score_complete[index:index+n_cycles][score_observed].std())
    dim = index*n_cycles
    print(index)
score_complete[:][0]
ssc = {}
ss = []
for dim in range(len(dimensions)):
    for i in range(n_cycles):
        ss.append()

#%%
from sklearn.model_selection import train_test_split

import csv

for count in range(5):

    analitic_dataset = pd.read_csv(path+'analitic_dataset_normalized.csv')
    x_train,x_val,Y_train,Y_val = train_test_split(analitic_dataset,np.array(atom_data.t_c),test_size=0.2)
    x_train,x_test,Y_train,Y_test = train_test_split(x_train,Y_train,test_size=0.2)


    simple_model = tf.keras.models.Sequential([tf.keras.layers.Dense(300,activation='relu'),
                                             tf.keras.layers.BatchNormalization(),
                                             tf.keras.layers.Dense(300,activation='relu'),
                                             tf.keras.layers.Dense(300,activation='relu'),
                                             tf.keras.layers.Dense(300,activation='relu'),
                                             tf.keras.layers.Dense(300,activation='relu'),
                                             tf.keras.layers.Dropout(0.3),
                                             tf.keras.layers.Dense(300,activation='relu'),
                                             tf.keras.layers.Dense(100,activation='relu'),
                                             tf.keras.layers.Dense(1,activation='linear')])

    simple_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001),
                      loss='mean_squared_error',
                      metrics=['mean_absolute_error'])


    early_stopping_callback = EarlyStopping(monitor = 'val_mean_absolute_error',min_delta = 0.05,patience = 5, restore_best_weights = True)
    simple_model.fit(x_train,Y_train,epochs= 30,batch_size=64,validation_data=(x_val,Y_val),callbacks=[early_stopping_callback])


    with open('simple_score.csv', mode='a') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csv_writer.writerow([simple_model.evaluate(x_test,Y_test,verbose=0)[1], simple_model.evaluate(x_test,Y_test,verbose = 0)[0], np.sqrt(simple_model.evaluate(x_test,Y_test,verbose = 0)[0])])

#%%
score_simple_model = pd.read_csv('simple_score.csv',header = None)
score_simple_model.tail()

punteggi_simple= {}

punteggi_simple[80] = 'MAE medio: {}; std: {}'.format(score_simple_model[:5][0].mean(),score_simple_model[:5][0].std())

punteggi_simple

#%%
from xgboost import XGBRegressor
gradient_model = XGBRegressor(n_estimators=100, learning_rate=0.01,max_depth= 4,gamma = 10)

eval_set = [(x_train, Y_train), (x_val, Y_val)]
eval_metric = ["mae","rmse"]
gradient_model.fit(x_train, Y_train,eval_metric=eval_metric, eval_set=eval_set, verbose=True,early_stopping_rounds =10)
gradient_y = gradient_model.predict(x_test)

tf.keras.losses.MAE(Y_test,gradient_y)
