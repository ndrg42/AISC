import sys
sys.path.append('/home/claudio/AISC/project_aisc/src/data')
sys.path.append('/home/claudio/AISC/project_aisc/src/features')
sys.path.append('/home/claudio/AISC/project_aisc/src/model')
import make_dataset
import build_features
import build_models
import numpy as np
import pandas as pd
import csv
import tensorflow as tf
import yaml
from yaml import Loader
import sklearn as sk
import importlib

def main():
    #Load atomic data
    ptable = make_dataset.PeriodicTable()
    #Initialize the processor for atomic data
    atom_processor = build_features.AtomData(ptable)
    #Process atomic data
    atom_processed = atom_processor.get_atom_data()

    #Load SuperCon dataset
    sc_dataframe = make_dataset.SuperCon(sc_path ='data/raw/supercon_tot.csv')
    #Initialize processor for SuperCon data
    supercon_processor = build_features.SuperConData(atom_processed,sc_dataframe,padding = 10)
    #Process SuperCon data
    supercon_processed = supercon_processor.get_dataset()

    tc_regression = sc_dataframe['critical_temp']

    file_model_config = open('/home/claudio/AISC/project_aisc/config/latent_dim_change_model_config.yaml')
    models_config = yaml.load_all(file_model_config,Loader)

    n_cycles = 1
    deep_set_score = {}
    for model_config in models_config:
        for i in range(n_cycles):
            X,X_test,Y,Y_test = build_features.train_test_split(supercon_processed,tc_regression,0.2)
            X,X_val,Y,Y_val = build_features.train_test_split(X,Y,0.2)

            #Define model and train it
            model = build_models.get_model(model_name='regressor',model_config = model_config)
            callbacks = [tf.keras.callbacks.EarlyStopping(min_delta=5,patience = 40,restore_best_weights=True)]
            model.fit(X,Y,validation_data=(X_val,Y_val),epochs=1,callbacks=callbacks)

            #Save scores and metrics' name
            deep_set_score[model_config['latent dim']] = model.evaluate(X_test,Y_test,verbose=0)

    file_model_config.close()
    n_cycles = 1
    supercon_processor.build_analitic_dataset()
    analytical_supercon_dataset_processed = supercon_processor.analytical_supercon_dataset_processed
    nn_score = {}
    for i in range(n_cycles):
        X,X_test,Y,Y_test = sk.model_selection.train_test_split(analytical_supercon_dataset_processed,tc_regression,test_size=0.2)
        X,X_val,Y,Y_val = sk.model_selection.train_test_split(X,Y,test_size = 0.2)

        #Define model and train it
        model = build_models.get_model(model_name='nn regressor',)
        callbacks = [tf.keras.callbacks.EarlyStopping(min_delta=5,patience = 40,restore_best_weights=True)]
        model.fit(X,Y,validation_data=(X_val,Y_val),epochs=1,callbacks=callbacks)

        #Save scores and metrics' name
        nn_score['80'] = model.evaluate(X_test,Y_test,verbose=0)
    print('\nDeep Set')
    for latent_dim in deep_set_score.keys():
        print(f'Latent dim:{latent_dim}  rmse: {deep_set_score[latent_dim][2]}')
    print('\nSimple Neural Network\n')
    for latent_dim in nn_score.keys():
        print(f'Latent dim:{latent_dim}  rmse: {nn_score[latent_dim][2]}')

if __name__ == '__main__':
    main()


# from xgboost import XGBRegressor
# gradient_model = XGBRegressor(n_estimators=100, learning_rate=0.01,max_depth= 4,gamma = 10)
#
# eval_set = [(x_train, Y_train), (x_val, Y_val)]
# eval_metric = ["mae","rmse"]
# gradient_model.fit(x_train, Y_train,eval_metric=eval_metric, eval_set=eval_set, verbose=True,early_stopping_rounds =10)
# gradient_y = gradient_model.predict(x_test)
#
# tf.keras.losses.MAE(Y_test,gradient_y)
