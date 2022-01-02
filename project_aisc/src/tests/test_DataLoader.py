"""module with test for DataLoader. Routine tested SuperCon

    SuperCon tests:
        type check for output: expected Pandas DataFrame
        correct format for dataframe: expected chemical symbols in the columns (up to atomic number =96)
        type check for critical temperature: expected float64
        check the correct formation of the dataset from the formula (string)

    PeriodicTable tests:
        type check for output: expected Pandas DataFrame
        check number of elements in the dataframe: expected chemical symbols 96
        equal refactored output with old alredy checked output: works if columns are 22/delete if you add other data

    CreateSuperCon tests:
        type check for output: expected Pandas DataFrame
        check number of elements in the dataframe: expected chemical symbols 96
        type check for critical temperature: expected float64
        check last 2 labels: expected ['critical_temp''material']
"""

#configuration setting for test
import os
os.getcwd()
import sys
sys.path.append('/home/claudio/AISC/project_aisc/src/data/')

#module needed for test or to test
import pytest
import pandas as pd
import numpy as np
import chela


from DataLoader import SuperCon
from DataLoader import PeriodicTable
from DataLoader import CreateSuperCon

test_path = 'data/raw/supercon_tot.csv'

#TEST on SuperCon------------------------------------------------------------------

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

#TEST on PeriodicTable-----------------------------------------------------------
#%%
#
# from DataLoader import from_string_to_dict,normalize_formula
#
# def CreateSuperCon(material=False,name='supercon_tot.csv',drop_heavy_element = False,normalized=False):
#     """Create a dataset of superconductor and non-superconducting materials
#
#     Args:
#         material (bool): a flag used to indicate if keep or not the material column
#         name (str): the name of the saved file (default is supercon_tot.csv)
#     """
#     #read data in a column separed format
#     supercon = pd.read_csv('data/raw/SuperCon_database.dat',delimiter = r"\s+",names = ['formula','tc'])
#     #supercon = pd.read_csv('data/raw/supercon_garbage.csv',names=['formula','tc'])
#     supercon.drop(0,inplace=True)
#     supercon['tc'] = supercon['tc'].astype(float)
#     #supercon.rename(columns = {'material':'formula','critical_temp':'tc'},inplace=True)
#     #remove rows with nan value on tc
#     supercon = supercon.dropna()
#     #get duplicated row aggregating them with mean value on critical temperature
#     duplicated_row = supercon[supercon.duplicated(subset = ['formula'],keep = False)].groupby('formula').aggregate({'tc':'mean'}).reset_index()
#     #drop all the duplicated row
#     supercon.drop_duplicates(subset = ['formula'],inplace = True,keep = False)
#     #compose the entire dataset
#     supercon= supercon.append(duplicated_row,ignore_index=True)
#     #initialize a dictionary with element symbol,critical_temp,material as keys
#     sc_dict={}
#     num_element = 96
#     for i in range(1,num_element+1):
#         sc_dict[element(i).symbol] = []
#
#     sc_dict['material'] = []
#     sc_dict['critical_temp'] = []
#     #list with all the element symbol
#     list_element = list(sc_dict.keys())[:num_element]
#     #search element that are put more than one time in a formula
#     repeted_values = []
#     list_heavy_element = list_element[86:96]
#     heavy_element = []
#     wrong_element = []
#     wrong_coef = []
#     zero_element = []
#     for i in range(supercon['formula'].shape[0]):
#
#         sc_string = supercon['formula'][i]
#         tupl_atom = []
#         from_string_to_dict(sc_string,tupl_atom)
#         list_atom = []
#         for j in range(len(tupl_atom)):
#             list_atom.append(tupl_atom[j][0])
#
#             if float(tupl_atom[j][1]) > 150:
#                 wrong_coef.append(i)
#
#
#             if float(tupl_atom[j][1]) == 0:
#                 zero_element.append(i)
#
#             if tupl_atom[j][0] not in list_element:
#                 wrong_element.append(i)
#
#             if tupl_atom[j][0] in list_heavy_element:
#                 heavy_element.append(i)
#                 break
#
#         if len(list(set(list_atom))) != len(list_atom):
#
#             repeted_values.append(i)
#     #drop repeted element and reset index
#     row_to_drop = []
#     if  drop_heavy_element:
#         row_to_drop = repeted_values + heavy_element + wrong_element + wrong_coef + zero_element
#         num_element = 86
#     else:
#         row_to_drop = repeted_values + wrong_element + wrong_coef+ zero_element
#         num_element = 96
#
#     row_to_drop = list(set(row_to_drop))
#     supercon.drop(row_to_drop,inplace = True)
#     # supercon.drop(repeted_values,inplace=True)
#     # supercon.drop(heavy_element,inplace=True)
#     # print(len(wrong_element))
#     # supercon.drop(wrong_element,inplace = True)
#     supercon.reset_index(drop= True, inplace=True)
#
#     sc_dict={}
#     #num_element = 86
#     for i in range(1,num_element+1):
#         sc_dict[element(i).symbol] = []
#
#     sc_dict['critical_temp'] = []
#     sc_dict['material'] = []
#
#
#     #list with the elements symbol
#     element_list = list(sc_dict.keys())
#     element_list = element_list[:-2]
#     element_list = set(element_list)
#     if normalized:
#         tupl_atom = normalize_formula(tupl_atom)
#     #create a dictionary with the quantity of each element on the molecules and relative chemical formula and critical temperature
#     for i in range(supercon['formula'].shape[0]):
#
#         sc_string = supercon['formula'][i]
#         sc_dict['material'].append(sc_string)
#         sc_dict['critical_temp'].append(float(supercon['tc'][i]))
#         tupl_atom = []
#         from_string_to_dict(sc_string,tupl_atom)
#         if normalized:
#             tupl_atom = normalize_formula(tupl_atom)
#         list_atom = []
#         for j in range(len(tupl_atom)):
#             list_atom.append(tupl_atom[j][0])
#
#             if tupl_atom[j][0] in list_element:
#                 sc_dict[tupl_atom[j][0]].append(float(tupl_atom[j][1]))
#
#         element_not_present = element_list - (set(list_atom))
#
#         for el in element_not_present:
#             sc_dict[el].append(0.0)
#
#     sc_dataframe = pd.DataFrame(sc_dict)
#     if not material:
#         sc_dataframe.drop(axis=1,inplace=True,columns=['material'])
#     return sc_dataframe
#
# #%%
# import chela
# chela.__version__
# import importlib
# import DataLoader
# importlib.reload(chela)
# from DataLoader import CreateSuperCon
#
# b = chela.csv_to_dataframe('data/raw/SuperCon_database.csv',header = True)
# a = DataLoader.CreateSuperCon(material=True)
#
# c = ((b.iloc[:,:96]- a.iloc[:,96])<0.01)
#
# c = a.iloc[:,:96]- b.iloc[:,:96]
#
# (c<0.01).all().all()
# c.all().all()
#
# a
# a.loc[:,['material','critical_temp']]#.to_csv('data/raw/SuperCon_database.csv',index = False)
# b
# b = b.drop([0])
# b.iloc[0:5,96:118].columns
# b = b.drop(axis = 0,columns = b.iloc[0:5,96:118].columns)
# c = b - a
# b = b.reset_index(drop=True)
# b =
#
# b = b.rename(columns = {'critical_temp':'tc','formula':'material'})
# a.loc[0,['material','K']]
#
# b
# b.loc[19140,['material','Zr','S','Se']]
#
# b['material']
# chela.from_string_to_dict(b.loc[0,'material'])
# b.loc[0,['material','K']]
#
# b.loc[:,['S','Se','K']]
# rand_int = np.random.randint(0,a.shape[0],10000)
# a.rename(columns = {'tc':'critical_temp','material':'formula'}).iloc[rand_int,:].drop(columns = ['critical_temp'])#.to_csv('10000_chemical_formulas_checked.csv',index=False)

#%%

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


#TEST on CreateSuperCon----------------------------------------------------------

test_path = 'data/raw/SuperCon_database.csv'

def test_create_supercon_type(test_path=test_path):
    """test create_supercon output type"""

    assert isinstance(CreateSuperCon(test_path), type(pd.DataFrame()))

def test_create_supercon_elements(test_path=test_path):
    """Check the presence of atomic symbols up to 96"""

    sc_dataset = CreateSuperCon(test_path)

    chemical_element = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

    #Last 2 columns are the chemical formulas and the critical temperature
    sc_chemical_elements = sc_dataset.columns[:-2]

    assert len(chemical_element[:96]) == len(sc_chemical_elements)

def test_create_supercon_critical_temperature(test_path=test_path):
    """"test critical temperature is a number"""

    assert CreateSuperCon(test_path).loc[:,'critical_temp'].dtype == 'float64'

def test_create_supercon_labels(test_path=test_path):
    """"Check the last 2 labels are material and critical temp for compatibility
        iusse with other part of the programm
    """
    last_two_names = CreateSuperCon(test_path).columns[-2:]

    assert (last_two_names == ['critical_temp','material']).all()
