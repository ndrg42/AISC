# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.experimental.set_synchronous_execution(False)


from tensorflow.keras.layers import Dense, Input, Add, Multiply
from tensorflow.keras.models import Model
import sys

sys.path.append('../data')
sys.path.append('../features')

import yaml
from yaml import Loader

with open('config/model_config.yaml') as file:
    model_config = yaml.load(file, Loader)


def build_phi(input, layers, output):
    """Return phi model of rho(sum_i phi(atom_i))"""

    input_atom = Input(shape=(input,))
    x = input_atom

    for layer in layers:
        x = Dense(**layer)(x)

    output = Dense(**output)(x)

    return Model(inputs=input_atom, outputs=output)


def build_rho(input, layers, output):
    """Return rho model of rho(sum_i phi(atom_i))"""

    atom_representation = Input(shape=(input,))
    x = atom_representation

    for layer in layers:
        x = Dense(**layer)(x)

    output = Dense(**output)(x)

    return Model(inputs=atom_representation, outputs=output)


class DeepSetModel(tf.keras.Model):
    """DeepSet model"""

    def __init__(self, phi_setup=model_config['phi setup'], rho_setup=model_config['regressor rho setup']):
        super(DeepSetModel, self).__init__()
        self.phi = build_phi(**phi_setup)
        self.rho = build_rho(**rho_setup)

    def call(self, atoms_input):
        phi_outputs = [self.phi(input) for input in atoms_input]
        material_representation = Add()(phi_outputs)
        rho_output = self.rho(material_representation)

        return rho_output


class LinearDeepSetModel(tf.keras.Model):
    """Linear Deep Set model"""

    def __init__(self, phi_setup=model_config['linear phi setup'], rho_setup=model_config['regressor rho setup']):
        super(LinearDeepSetModel, self).__init__()
        self.phi = build_phi(**phi_setup)
        self.rho = build_rho(**rho_setup)

    def call(self, atoms_input):
        phi_outputs = [Multiply()([tf.expand_dims(input[:, -1], 1), self.phi(input[:, :-1])]) for input in atoms_input]
        material_representation = Add()(phi_outputs)
        rho_output = self.rho(material_representation)

        return rho_output

    def material_representation(self, atoms_input):
        phi_outputs = [Multiply()([tf.expand_dims(input[:, -1], 1), self.phi(input[:, :-1])]) for input in atoms_input]
        material_output = Add()(phi_outputs)

        return material_output

    def atom_representation(self, atoms_input):
        atom_output = [self.phi(input[:, :-1]) for input in atoms_input][0]

        return atom_output


def build_vanilla_nn(input, layers, output):
    """Vanilla neural network that takes as input an engineered features input"""

    model = tf.keras.Sequential()
    model.add(Dense(**input))

    for layer in layers:
        model.add(Dense(**layer))

    model.add(Dense(**output))

    return model


def get_vanilla_nn_regressor(nn_setup=model_config['neural network setup'],
                             regressor_setup=model_config['regressor neural network setup'],
                             ):
    model = build_vanilla_nn(**nn_setup)
    model.compile(optimizer=regressor_setup['optimizer'](regressor_setup['learning rate']),
                  loss=regressor_setup['loss'],
                  metrics=[metric if isinstance(metric, str) else metric() for metric in regressor_setup['metrics']],
                  )
    return model


def get_linear_deepset_regressor(phi_setup=model_config['linear phi setup'],
                                 rho_setup=model_config['regressor rho setup'],
                                 regressor_setup=model_config['regressor setup'],
                                 ):
    linear_regressor_deepset = LinearDeepSetModel(phi_setup, rho_setup)
    linear_regressor_deepset.compile(optimizer=regressor_setup['optimizer'](regressor_setup['learning rate']),
                                     loss=regressor_setup['loss'],
                                     metrics=[metric if isinstance(metric, str) else metric() for metric in
                                              regressor_setup['metrics']],
                                     )
    return linear_regressor_deepset


def get_linear_deepset_classifier(phi_setup=model_config['linear phi setup'],
                                  rho_setup=model_config['classifier rho setup'],
                                  classifier_setup=model_config['classifier setup'],
                                  ):
    classifier_deepset = LinearDeepSetModel(phi_setup, rho_setup)
    classifier_deepset.compile(optimizer=classifier_setup['optimizer'](classifier_setup['learning rate']),
                               loss=classifier_setup['loss'],
                               metrics=[metric if isinstance(metric, str) else metric() for metric in
                                        classifier_setup['metrics']],
                               )
    return classifier_deepset


def get_deepset_regressor(phi_setup=model_config['phi setup'],
                          rho_setup=model_config['regressor rho setup'],
                          regressor_setup=model_config['regressor setup'],
                          ):
    regressor_deepset = DeepSetModel(phi_setup, rho_setup)
    regressor_deepset.compile(optimizer=regressor_setup['optimizer'](regressor_setup['learning rate']),
                              loss=regressor_setup['loss'],
                              metrics=[metric if isinstance(metric, str) else metric() for metric in
                                       regressor_setup['metrics']],
                              )

    return regressor_deepset


def get_deepset_classifier(phi_setup=model_config['phi setup'],
                           rho_setup=model_config['classifier rho setup'],
                           classifier_setup=model_config['classifier setup'],
                           ):
    classifier_deepset = DeepSetModel(phi_setup, rho_setup)
    classifier_deepset.compile(optimizer=classifier_setup['optimizer'](classifier_setup['learning rate']),
                               loss=classifier_setup['loss'],
                               metrics=[metric if isinstance(metric, str) else metric() for metric in
                                        classifier_setup['metrics']],
                               )
    return classifier_deepset


with open('config/available_model_config.yaml') as file:
    # Load a dictionary containing the model's name and the function to initialize them
    avaible_model = yaml.load(file, Loader)


def get_model(model_name='classifier', model_config=None):
    """Retrieve and return the specified model using available_model (dict) as a switch control."""

    model_builder = avaible_model.get(model_name)

    if model_config is None:
        try:
            return model_builder()
        except:
            print('Model not found.')
    else:
        try:
            return model_builder(*model_config_initializer(model_config=model_config, model_name=model_name))
        except:
            print('Model not found.')


def model_config_initializer(model_config, model_name):
    if 'nn' in model_name:
        return model_config['neural network setup'], model_config['regressor neural network setup']
    if 'linear' in model_name:
        model_name = model_name.replace('linear ', '')
        return model_config['linear phi setup'], model_config[model_name + ' rho setup'], model_config[
            model_name + ' setup']
    else:
        return model_config['phi setup'], model_config[model_name + ' rho setup'], model_config[model_name + ' setup']
