import sys
sys.path.append('/home/claudio/AISC/project_aisc/src/features/')
sys.path.append('/home/claudio/AISC/project_aisc/src/data/')

#module needed for test or to test
import pytest
import pandas as pd
import numpy as np
import chela

import DataLoader
import Processing
import importlib
importlib.reload(Processing)

#
# ptable = DataLoader.PeriodicTable()
# sc_dataframe = DataLoader.SuperCon(sc_path ='data/raw/supercon_tot.csv')
#
# atom_data = Processing.DataProcessor(ptable, sc_dataframe)
# atom_data.build_Atom()
#
# ((atom_data.Atom - atom_data_checked)<0.001).all().all()
# old_atom_data = atom_data.Atom
#
# atom_data_checked = atom_data_checked.rename(columns={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10})
#
# atom_data_checked = pd.read_csv('data/processed/dataset_atom.csv',index_col=0)
#
#
# test_path = 'data/raw/supercon_tot.csv'
def test_build_atom_data():

    atom_data_checked = pd.read_csv('data/processed/dataset_atom.csv',index_col=0)
    atom_data_checked = atom_data_checked.rename(columns={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10})

    ptable = DataLoader.PeriodicTable()
    atom_data = Processing.AtomData(ptable)
    atom_data.build_atom_data()

    assert ((atom_data.atom_data.drop(columns = ['group_id']) - atom_data_checked.drop(columns = ['group_id']))<0.001).all().all()
