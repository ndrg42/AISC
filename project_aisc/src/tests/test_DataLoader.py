"""module with test for DataLoader. Routine tested SuperCon

    SuperCon test:
        type check for output: expected Pandas DataFrame
        correct format for dataframe: expected chemical symbols in the columns (up to atomic number =96)
        type check for critical temperature: expected float64
        check the correct formation of the dataset from the formula (string)

    PeriodicTable test:
        type check for output: expected Pandas DataFrame
        correct format for dataframe: expected chemical symbols in the columns (up to atomic number =96)

"""

#configuration setting for test
import os
os.getcwd()
import sys
sys.path.append('src/data')

#module needed for test or to test
import pytest
import pandas as pd
import numpy as np
import chela


from DataLoader import SuperCon
from DataLoader import PeriodicTable

test_path = 'data/raw/supercon_tot.csv'

def test_supercon_type(test_path=test_path):
    """test supercon output type"""

    assert isinstance(SuperCon(test_path), type(pd.DataFrame()))


def test_supercon_chemical_elements(test_path = test_path):
    """test the presence of all(up to 96) chemical elements (symbols) as columns"""

    chemical_element = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

    supercon_columns = list(SuperCon(test_path).columns)

    for column_name_not_chemical_symbols in ['material','critical_temp']:
        try:
            supercon_columns.remove(column_name_not_chemical_symbols)
        except:
            pass

    assert supercon_columns == chemical_element[:96]

def test_supercon_critical_temperature(test_path=test_path):
    """"test critical temperature is a number"""

    assert SuperCon(test_path).loc[:,'critical_temp'].dtype == 'float64'

#useful if you use a dataset alredy done and you want to check the correctness
#of the transformation string -> dataset for a sample of chemical formulas
#chela has alredy its tests
def test_supercon_correct_splitted_formula_on_columns(test_path=test_path,number_of_rows=50):
    """test the transformation string to a dataframe for number_of_rows formulas"""

    chemical_element = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

    super_con = SuperCon(test_path)
    random_int = np.random.randint(0,super_con.shape[0],number_of_rows)

    converted_formulas = pd.DataFrame([chela.from_string_to_dict(super_con.loc[random_formula,'material']) for random_formula in random_int ],columns = chemical_element[:96])
    converted_formulas = converted_formulas.replace(np.nan,0)

    assert ((super_con.iloc[random_int,:-2].reset_index(drop=True) - converted_formulas)<0.001).all(axis=1).all()

# #%%
# import mendeleev
#
# periodic_table = mendeleev.get_table('elements')
# periodic_table.shape
# col_vuote = []
# mancanti = periodic_table.isna().sum()
# max_missing_value = 30
# for i in range(mancanti.size):
#     if mancanti[i] >= max_missing_value:
#         col_vuote.append(mancanti.index[i])
#
# col_vuote
# periodic_table.columns
# col_vuote.remove('thermal_conductivity')
# col_vuote.remove('fusion_heat')
# col_vuote.remove('electron_affinity')
# periodic_table.drop(col_vuote,axis = 1,inplace=True)
# periodic_table = periodic_table.iloc[:96,:]
#
#
#
# path = "data/raw/"
# thermal_conductivity = pd.read_csv(path+"LatticeConstants.csv",header=None)
#
# thermal_conductivity.replace('QuantityMagnitude[Missing["NotAvailable"]]',np.nan,inplace=True)
# thermal_conductivity.replace('QuantityMagnitude[Missing["Unknown"]]',np.nan,inplace=True)
#
# thermal_conductivity_periodic_table = periodic_table.loc[:,'fusion_heat']
# thermal_conductivity
#
# thermal_conductivity_periodic_table.isna().sum()
#
# thermal_conductivity[thermal_conductivity[0].isna()]
# thermal_conductivity_periodic_table
# thermal_conductivity_periodic_table[thermal_conductivity_periodic_table['fusion_heat'].isna()]
#
# thermal_conductivity_periodic_table.isna().sum()
# thermal_conductivity.isna().sum()
#
# thermal_conductivity_periodic_table.fillna(thermal_conductivity[0])
#
# periodic_table
# #%%
#
# periodic_table = mendeleev.get_table('elements')
# max_index_atom = 109
# max_missing_value = 30
# #PROVA: TOLGO "en_pauling","group_id","evaporation_heat"
# atomic_features_to_drop = ['annotation','description','name','jmol_color','symbol','is_radioactive','vdw_radius_mm3',
#                        'cpk_color','uses','sources','name_origin','discovery_location','covalent_radius_cordero',
#                        'discoverers','cas','goldschmidt_class','molcas_gv_color','discovery_year','atomic_radius','series_id',
#                        'electronic_configuration','glawe_number','en_ghosh','heat_of_formation','covalent_radius_pyykko_double',
#                        'vdw_radius_alvarez','abundance_crust', 'abundance_sea', 'c6_gb','vdw_radius_uff',
#                        'dipole_polarizability_unc','boiling_point','pettifor_number','mendeleev_number']
#
# #There isn't data for elements heavier than 109
# ionenergies_col = [element(index).ionenergies[1] for index in range(1,max_index_atom)]
# valence_col = [element(index).nvalence() for index in range(1,max_index_atom)]
#
# periodic_table = periodic_table.drop(atomic_features_to_drop,axis = 1)
# periodic_table = periodic_table[:(max_index_atom-1)]
#
# periodic_table['valence'] = valence_col
# periodic_table['ionenergies'] = ionenergies_col
# periodic_table = periodic_table[:96]
# #
# #
#
# dataset = periodic_table
# empty_columns = [column for column in dataset.columns if dataset[column].isna().sum()> max_missing_value]
# empty_columns
#
# exceptions
# #remove from the list exceptional columns even if they have too many missing values
# if exceptions:
#     for column in exceptions:
#         if column in empty_columns:
#             empty_columns.remove(column)
#
# empty_columns
# import importlib
# importlib.reload(DataLoader)
#
# DataLoader.PeriodicTable().isna().sum()
#
# geochemical_class
# covalent_radius_pyykko_triple
# en_allen
#
# periodic_table_correct = pd.read_csv('data/processed/periodic_table.csv',index_col=0)
# periodic_table_numeric_correct = periodic_table_correct.select_dtypes(include=numerics).replace(np.nan,0)
# periodic_table_correct.isna().sum()

# periodic_table.shape
# col_vuote = []
# mancanti = periodic_table.isna().sum()
# for i in range(mancanti.size):
#     if mancanti[i] >= max_missing_value:
#         col_vuote.append(mancanti.index[i])
#
# dataset['en_allen'].isna().sum()
# mancanti
# empyt_columns
#
#
# dataset= periodic_table
# dataset['fusion_heat'].isna().sum()
# empyt_columns = [column for column in dataset.columns if dataset[column].isna().sum()> max_missing_value]
# empyt_columns = [column for column in dataset.columns if dataset[column].isna().sum()]
#
# periodic_table['thermal_conductivity'].isna().sum()
#
# empyt_columns
#
# def remove_columns_with_missing_elements(dataset = dataset,max_missing_value = 30, exceptions= exceptions):
#    """remove columns that has more than max_missing_value with expection for except columns"""
#
#    empty_columns = [column for column in dataset.columns if dataset[column].isna().sum()> max_missing_value]
#
#    #remove from the list exceptional columns even if they have too much missing values
#    if exceptions:
#        for column in exceptions:
#            empty_columns.remove(column)
#
#     return dataset.drop(empty_columns,axis = 1)
#
#
#
# col_vuote
# col_vuote.remove('thermal_conductivity')
# col_vuote.remove('fusion_heat')
# col_vuote.remove('electron_affinity')
# periodic_table.drop(col_vuote,axis = 1,inplace=True)
# periodic_table = periodic_table[:96]
#
# periodic_table['specific_heat']#.isna().sum()
#
# thermal_conductivity[2].iloc[6].astype('float64')/100
#DataLoader.PeriodicTable()
#%%
# numerics = ['int16','int32','int64','float16','float32','float64']
#
# periodic_table_correct = pd.read_csv('data/processed/periodic_table.csv',index_col=0)
# periodic_table_numeric_correct = periodic_table_correct.select_dtypes(include=numerics).replace(np.nan,0)
#
# periodic_table_numeric = DataLoader.PeriodicTable().select_dtypes(include=numerics).replace(np.nan,0)
#
# ((periodic_table_numeric_correct - periodic_table_numeric) <0.001).all(axis=1).all()
#
# periodic_table_numeric

#%%
import importlib
#importlib.reload(DataLoader)

import DataLoader

def test_periodic_table_type():
    """test periodic table  output type"""

    assert isinstance(PeriodicTable(), type(pd.DataFrame()))

def test_periodic_table_numeric_output():
    """test periodic table format with one alredy controlled"""

    numerics = ['int16','int32','int64','float16','float32','float64']

    periodic_table_correct = pd.read_csv('data/processed/periodic_table.csv',index_col=0)
    periodic_table_numeric_correct = periodic_table_correct.select_dtypes(include=numerics).replace(np.nan,0)

    periodic_table_numeric = PeriodicTable().select_dtypes(include=numerics).replace(np.nan,0)

    assert ((periodic_table_numeric_correct - periodic_table_numeric) <0.01).all(axis=1).all()


def test_number_rows_periodic_table():
    """Check number of rows is equal to max_atomic_number (96)"""

    number_of_rows = PeriodicTable().shape[0]

    assert number_of_rows == 96
