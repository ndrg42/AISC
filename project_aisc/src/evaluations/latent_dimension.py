import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.append('/home/claudio/AISC/project_aisc/src/data')
sys.path.append('/home/claudio/AISC/project_aisc/src/features')
sys.path.append('/home/claudio/AISC/project_aisc/src/model')
import make_dataset
import build_features
import build_models
import tensorflow as tf
import yaml
from yaml import Loader
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import mendeleev

def latent_parser():

    with open('/home/claudio/AISC/project_aisc/config/avaible_model_config.yaml') as file:
        model_config = yaml.load(file,Loader)


    my_parser = argparse.ArgumentParser(prog='latent_dimension',
                                        description = "train and display latent space of linear model",
                                        usage = '%(prog)s [options]',
                                        )
    my_parser.add_argument('-model',
                          action = 'store',
                          nargs = '+',
                          metavar = ('MODEL'),
                          help = """Select the linear model. Possible choice:
                                  -linear regressor
                                  -linear classifier"""
    )

    my_parser.add_argument('-config',
                          action = 'store',
                          nargs = 1,
                          metavar = ('CONFIG'),
                          help = """Use a custom config for the ML model.
                                   The model need to be specified"""
    )

    #Parse the args
    args = my_parser.parse_args()

    return args


def main():

    with open('/home/claudio/AISC/project_aisc/config/latent_dimension_config.yaml') as file:
        model_config = yaml.load(file,Loader)

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
    model_config_path = None

    if len_argv > 1:
        args = latent_parser()
        model_name = ' '.join(args.model)
        model_config_path = args.config

    else:
        model_name = 'linear regressor'

    if 'regressor' in model_name:
        tc = sc_dataframe['critical_temp']
        #Select material with high temperature (tc > 10 K)
        mask_temperature_materials = np.where(tc > 10,1,0)
        legend_latent_space = ['High temperature','Low temperature']

    elif 'classifier' in model_name:
        tc = np.where(sc_dataframe['critical_temp']>0,1,0)
        mask_temperature_materials = tc
        legend_latent_space = ['Superconductor','Non Superconductor']

    if model_config_path is not None:
        model_config_path = model_config_path[0]

        with open(model_config_path) as file:
            model_config = yaml.load(file,Loader)


    X,X_test,Y,Y_test = build_features.train_test_split(supercon_processed, tc, 0.2)
    X,X_val,Y,Y_val = build_features.train_test_split(X, Y, 0.2)

    #Define model and train it
    model = build_models.get_model(model_name = model_name ,model_config = model_config)
    callbacks = [tf.keras.callbacks.EarlyStopping(min_delta = 5, patience = 40, restore_best_weights = True)]
    model.fit(X, Y, validation_data = (X_val,Y_val), epochs = 1, callbacks = callbacks)


    materials_representation = model.material_representation(supercon_processed)

    elements = [mendeleev.element(i).symbol for i in range(1,97)]

    elements_dataframe = pd.DataFrame(np.eye(96),columns = elements)
    atoms_processor = build_features.SuperConData(atom_processed,elements_dataframe)
    #Process SuperCon data
    atoms_processed = atoms_processor.get_dataset()

    atoms_representation = pd.DataFrame(model.atom_representation(atoms_processed).numpy(),index=elements,columns=['latent feature'])
    fig,axs = plt.subplots(1,2, figsize=(11,8))

    sns.histplot(x = np.reshape(materials_representation.numpy(),(-1,)), hue = mask_temperature_materials, ax = axs[0])
    axs[0].set_title('Material Latent Representation')
    axs[0].set_xlabel('Latent feature')
    axs[0].legend(legend_latent_space)

    sns.scatterplot(data = atoms_representation,x = 'latent feature', y = range(1,97), ax = axs[1])
    axs[1].set_title('Atomic Latent Representation')
    axs[1].set_xlabel('Latent feature')
    axs[1].set_ylabel('Atomic number')
    axs[1].set_yticks([])

    for line in range(0,atoms_representation.shape[0]):
         axs[1].text(x = atoms_representation.iloc[line] + 0.02,
                     y = line+1,
                     s = elements[line],
                     fontsize = 8,
                    )


    plt.show()
    #print(pd.DataFrame(model.atom_representation(atoms_processed).numpy(),index=elements,columns = ['Latent representation']).to_string())
    print(atoms_representation)

if __name__ == '__main__':
    main()
