import sys
sys.path.append('src/data')
sys.path.append('src/features')
sys.path.append('src/model')
import make_dataset
import build_features
import build_models
import tensorflow as tf
import argparse
import yaml
from yaml import Loader
import numpy as np

def train_parser():

    with open('/home/claudio/AISC/project_aisc/config/avaible_model_config.yaml') as file:
        model_config = yaml.load(file,Loader)


    my_parser = argparse.ArgumentParser(prog='train model',
                                        description = "train model on superconduttivity data",
                                        usage = '%(prog)s [options]',
                                        )
    my_parser.add_argument('-model',
                          action = 'store',
                          nargs = '+',
                          metavar = ('MODEL'),
                          help = """Select the model. Possible choice:
                                  - """ +'\n-'.join(model_config.keys())
    )

    my_parser.add_argument('-config',
                          action = 'store',
                          nargs = 1,
                          metavar = ('CONFIG'),
                          help = """Use a custom config for the ML model.
                                   The model need to be specified"""
    )

    my_parser.add_argument('-save',
                           action = 'store',
                           nargs = '?',
                           help = "Save the results into a folder ",
    )

    #Parse the args
    args = my_parser.parse_args()

    return args

def save_results():
    pass


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

    len_argv = len(sys.argv)
    model_config = None
    model_config_path = None

    if len_argv > 1:
        args = train_parser()
        model_name = ' '.join(args.model)
        model_config_path = args.config

    else:
        model_name = 'regressor'

    if 'regressor' in model_name:
        tc = sc_dataframe['critical_temp']

    elif 'classifier' in model_name:
        tc = np.where(sc_dataframe['critical_temp']>0,1,0)

    if model_config_path is not None:
        model_config_path = model_config_path[0]

        with open(model_config_path) as file:
            model_config = yaml.load(file,Loader)


    X,X_test,Y,Y_test = build_features.train_test_split(supercon_processed, tc, 0.2)
    X,X_val,Y,Y_val = build_features.train_test_split(X,Y,0.2)

    #Define model and train it
    model = build_models.get_model(model_name= model_name, model_config = model_config)
    callbacks = [tf.keras.callbacks.EarlyStopping(min_delta=5,patience = 40,restore_best_weights=True)]
    model.fit(X,Y,validation_data=(X_val,Y_val),epochs=2,callbacks=callbacks)

    #Save scores and metrics' name
    score = model.evaluate(X_test,Y_test,verbose=0)
    metrics_name = [metric.name for metric in model.metrics]
    #Print the metric and the relative score of the model
    for name,value in zip(metrics_name,score):
        print(name+':',value)

if __name__ == '__main__':
    main()
