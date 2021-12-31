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


import importlib
import DataLoader
#     for i in range(1,max_index_atom):
#         el = element(i)
#         el = el.ionenergies[1]
#         ionenergies_col.append(el)
#     for i in range(1,max_index_atom):
#         el = element(i)
#         el = el.nvalence()
#         valence_col.append(el)
#
# from mendeleev import  element
#element(1).ionenergies[1]
#Periodic Table test:
#    output check type
#    columns name
#    atomic number int
#    atomic volume > 0
#    block string
#    period int
#    valence int
#    group id int

def test_periodic_table_type():
    """test periodic table  output type"""

    assert isinstance(PeriodicTable(), type(pd.DataFrame()))

def test_periodic_table_numeric_output():
    """test periodic table format with one alredy controlled"""

    numerics = ['int16','int32','int64','float16','float32','float64']

    periodic_table_correct = pd.read_csv('data/processed/periodic_table.csv',index_col=0)
    periodic_table_numeric_correct = periodic_table_correct.select_dtypes(include=numerics).replace(np.nan,0)

    periodic_table_numeric = PeriodicTable().select_dtypes(include=numerics).replace(np.nan,0)

    assert ((periodic_table_numeric_correct - periodic_table_numeric) <0.001).all(axis=1).all()
