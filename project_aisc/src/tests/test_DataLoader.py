#configuration setting for test
import os
os.getcwd()
import sys
sys.path.append('src/data')

#module needed for test or to test
import pytest
import pandas as pd

from DataLoader import SuperCon
#TO DO list for DataLoader:
#is a Dataframe?-> right type
#is in the correct format? -> chemical elements, formula,critical temperature

test_path = 'data/raw/supercon_tot.csv'

def test_supercon_type(test_path=test_path):
    """test supercon output type"""

    assert isinstance(SuperCon(test_path), type(pd.DataFrame()))


def test_supercon_chemical_elements(test_path = test_path):
    """test the presence of all(up to 96) chemical elements (symbols) as columns"""

    chemical_element = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

    supercon_columns = list(SuperCon(test_path).columns)

    for column_name_not_atom in ['material','critical_temp']:
        try:
            supercon_columns.remove(column_name_not_atom)
        except:
            pass

    assert supercon_columns == chemical_element[:96]
