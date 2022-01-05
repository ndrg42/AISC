import sys
sys.path.append('/home/claudio/AISC/project_aisc/src/features/')
sys.path.append('/home/claudio/AISC/project_aisc/src/data/')
sys.path.append('/home/claudio/AISC/project_aisc/src/model/')
#module needed for test or to test
import pytest
import pandas as pd
import numpy as np
import chela

import DataLoader
import Processing
import DeepSets
import importlib
importlib.reload(DeepSets)

@pytest.fixture
def supercon_dataset():
    return list(np.load('data/processed/dataset_supercon.npy'))


def test_regressor_deepset_build(supercon_dataset):
    """Check if model definitions work"""
    regressor_deepset = DeepSets.RegressorDeepSet(supercon_dataset)
    regressor_deepset.build_model()

    assert regressor_deepset.rho != None and regressor_deepset.phi != None

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
