import sys
sys.path.append('/home/claudio/AISC/project_aisc/src/features/')
sys.path.append('/home/claudio/AISC/project_aisc/src/data/')
sys.path.append('/home/claudio/AISC/project_aisc/src/model/')
#module needed for test or to test
import pytest
import pandas as pd
import numpy as np
import chela
from sklearn.model_selection import train_test_split
import DataLoader
import Processing
import DeepSets
import importlib
importlib.reload(DeepSets)

@pytest.fixture
def supercon_dataset():
    return list(np.load('data/processed/dataset_supercon.npy'))

@pytest.fixture
def supercon_temperature():
    return DataLoader.SuperCon(sc_path ='data/raw/supercon_tot.csv')['critical_temp']


def test_regressor_deepset_build(supercon_dataset):
    """Check if model definition work"""
    regressor_deepset = DeepSets.RegressorDeepSet(supercon_dataset)
    regressor_deepset.build_model()

    assert regressor_deepset.rho != None and regressor_deepset.phi != None

def test_deepset_train(supercon_dataset,supercon_temperature):
    """Test method for model fitting"""

    X,X_val,y,y_val = train_test_split(np.moveaxis(np.array(supercon_dataset),0,1),np.array(supercon_temperature),test_size = 0.3)
    X  = list(np.moveaxis(X,0,1))
    X_val  = list(np.moveaxis(X_val,0,1))

    regressor_deepset = DeepSets.RegressorDeepSet(supercon_dataset)
    regressor_deepset.build_model()
    regressor_deepset.fit_model(X=X,y=y,X_val=X_val,y_val=y_val,epochs=1)

    assert regressor_deepset.history != None


#
# sc_dataframe = DataLoader.SuperCon(sc_path ='data/raw/supercon_tot.csv')
# supercon_temperature = sc_dataframe['critical_temp']
# supercon_dataset = list(np.load('data/processed/dataset_supercon.npy'))
#
# regressor_deepset = DeepSets.RegressorDeepSet(supercon_dataset)
#
# regressor_deepset != None
#
# isinstance(regressor_deepset,type(DeepSets.RegressorDeepSet(supercon_dataset)))
#
#
# regressor_deepset.rho != None
#
# regressor_deepset.build_model()
# regressor_deepset.rho != None
