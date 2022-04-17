
import pandas as pd
import numpy as np
from mendeleev import element
from mendeleev import get_table
import sys
sys.path.append('src/features')
import make_dataset
from Processing import remove_columns_with_missing_elements


#make_dataset.py rename into make_dataset.py
#deve contenere solo routine per generare/scaricare/caricare nel giusto formato
#i dati raw

#SuperCon -> Da cambiare nome
#PeriodicTable ->Da cambiare nome
#CreateSuperCon -> Da cambiare nome
#from_string_to_dict -> da cancellare, non serve più. è spostato nel pacchetto chela

#Change name according python standard
def SuperCon(sc_path = '../data/raw/supercon_tot.csv'):
    """
    carica un dataset con gli atomi che compongono i materiali superconduttori e la loro temperatura critica

    output:
          dataset
    """

    sc_dataframe = pd.read_csv(sc_path)
    return sc_dataframe


def PeriodicTable(max_atomic_number=96,max_missing_value=30):
    """
    carica i dataset dei composti superconduttori e la tavola periodica
    features atomiche disponibili:'atomic_number', 'atomic_volume', 'block', 'density',
       'dipole_polarizability', 'electron_affinity', 'evaporation_heat',
       'fusion_heat', 'group_id', 'lattice_constant', 'lattice_structure',
       'melting_point', 'period', 'specific_heat', 'thermal_conductivity',
       'vdw_radius', 'covalent_radius_pyykko', 'en_pauling', 'atomic_weight',
       'atomic_radius_rahm', 'valence', 'ionenergies'
    input:
          numero massimo di atomi da caricare (opzionale)
          numero massimo di dati mancanti per feature (opzionale)
    output:
          dataset
    """

    periodic_table = get_mendeleev_periodic_table_data(max_atomic_number=max_atomic_number)


    exceptions = ['thermal_conductivity','fusion_heat','electron_affinity','specific_heat']
    periodic_table = remove_columns_with_missing_elements(dataset = periodic_table,
                                                          exceptions = exceptions,
                                                          max_missing_value=max_missing_value,
                                                          )

    #../../
    path = "data/raw/"
    features = ['thermal_conductivity','specific_heat','electron_affinity','density']
    features_and_scale = {'thermal_conductivity':1,'specific_heat':1/1000,'electron_affinity': 1/100,'density':1/1000}

    atomic_dataset_dict = get_external_periodic_table_data(path = path, features = features)


    periodic_table = merge_periodic_table_data(features_and_scale=features_and_scale,
                                               atomic_dataset_dict= atomic_dataset_dict,
                                               periodic_table= periodic_table)

    return periodic_table


def get_mendeleev_periodic_table_data(max_atomic_number=96):
    """get periodic table from mendeleev software up to atomic number = max_atomic_number"""

    periodic_table = get_table('elements')
    periodic_table = periodic_table.iloc[:max_atomic_number,:]


    #Atomic features we don't  think are linked to superconductivity
    atomic_features_to_drop = ['annotation','description','name','jmol_color','symbol','is_radioactive','vdw_radius_mm3',
                           'cpk_color','uses','sources','name_origin','discovery_location','covalent_radius_cordero',
                           'discoverers','cas','goldschmidt_class','molcas_gv_color','discovery_year','atomic_radius','series_id',
                           'electronic_configuration','glawe_number','en_ghosh','heat_of_formation','covalent_radius_pyykko_double',
                           'vdw_radius_alvarez','abundance_crust', 'abundance_sea', 'c6_gb','vdw_radius_uff',
                           'dipole_polarizability_unc','boiling_point','pettifor_number','mendeleev_number','geochemical_class',
                           'covalent_radius_pyykko_triple', 'en_allen','atomic_weight_uncertainty']

    #There isn't data for elements heavier than (Z)109 however we restrinc the analisys to 96
    #ionenergies and valence are not presence in get_table so we add them with another function
    ionenergies_column = [element(index).ionenergies[1] for index in range(1,max_atomic_number+1)]
    valence_column = [element(index).nvalence() for index in range(1,max_atomic_number+1)]

    periodic_table = periodic_table.drop(atomic_features_to_drop,axis = 1)


    periodic_table['valence'] = valence_column
    periodic_table['ionenergies'] = ionenergies_column

    return periodic_table


def get_external_periodic_table_data(path,features):
    """Load a list of pandas DataFrame containing single atomic features"""

    #These Dataset are meant to be merged with periodic table data from mendeleev software
    #They have no header and contain only a single atomic feature
    #There is a specific clean process where we replace some strings (they indicate no avaible data) with Nan value
    atomic_dataset_list = [pd.read_csv(path+feature+'.csv',header=0) for feature in features]

    atomic_dataset_list = list(map(lambda x: x.replace('QuantityMagnitude[Missing["NotAvailable"]]',np.nan),atomic_dataset_list))
    atomic_dataset_list = list(map(lambda x: x.replace('QuantityMagnitude[Missing["Unknown"]]',np.nan),atomic_dataset_list))

    #turn the list into a dictionary to be more manageable
    atomic_dataset_dict = {feature:atomic_dataset_list[index] for index,feature in enumerate(features)}

    return atomic_dataset_dict


def merge_periodic_table_data(features_and_scale,atomic_dataset_dict,periodic_table):
    """Merge periodic table from get_external_periodic_table_data (and scale them) and mendeleev data"""

    for feature in features_and_scale.keys():
        scaled_feature = atomic_dataset_dict[feature][feature].astype('float32')*features_and_scale[feature]
        periodic_table[feature] = periodic_table[feature].fillna(scaled_feature)

    return periodic_table


# def CreateSuperCon(path,material=True,name='supercon_tot.csv',drop_heavy_element = False,normalized=False):
#     """Create a dataset of superconductor and non-superconducting materials
#
#     Args:
#         material (bool): a flag used to indicate if keep or not the material column
#         name (str): the name of the saved file (default is supercon_tot.csv)
#     """
#     #read data in a column separed format
#     #supercon = pd.read_csv('data/raw/SuperCon_database.dat',delimiter = r"\s+",names = ['formula','tc'])
#     supercon = pd.read_csv(path)
#     supercon = supercon.rename(columns = {'critical_temp':'tc','material':'formula'})
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
#     #sc_dataframe.to_csv('../../data/raw/'+name,index = False)
from chela import csv_to_dataframe

def CreateSuperCon(path,material=True,name='supercon_tot.csv',drop_heavy_element = False,normalized=False):
    """Create a dataset of superconductor and non-superconducting materials

    Args:
        material (bool): a flag used to indicate if keep or not the material column
        name (str): the name of the saved file (default is supercon_tot.csv)
    """

    supercon = csv_to_dataframe(path)
    #We use material as label for chemical formulas
    supercon = supercon.rename(columns = {'formula':'material'})
    #chela.csv_to_dataframe create a dataframe with 1-118 chemical elements
    #but we want only the first 96
    chemical_elements_to_drop = supercon.columns[96:118]
    supercon = supercon.drop(columns = chemical_elements_to_drop)
    #We swap the last 2 columns because we want 'material' as last one
    chemical_symbols = supercon.columns[:96]
    chemical_symbols = list(chemical_symbols)
    chemical_symbols.append('critical_temp')
    chemical_symbols.append('material')
    supercon = supercon[chemical_symbols]

    return supercon




#
# def from_string_to_dict(string,lista):
#     """add to a list tuples containing elements and the relative quantity presence
#
#     Args:
#         string (str): string of the material
#         lista (list): list where the couples element, quantity are added
#     """
#     nums = ['0','1','2','3','4','5','6','7','8','9','.']
#     i = 0
#     element_name = ''
#     element_quantity = ''
#     on = True
#
#     while(i<len(string) and on ):
#         if string[i] not in nums:
#
#             element_name = element_name + string[i]
#
#             if i == len(string)-1:
#                 lista.append((element_name,'1'))
#                 return
#             if i+1 < len(string):
#                 if string[i+1].isupper() :
#                     lista.append((element_name,'1'))
#                     string = string[i+1:]
#                     from_string_to_dict(string,lista)
#                     return
#
#
#         if string[i] in nums:
#             element_quantity = ''
#             for j in range(len(string)-i):
#
#                 if string[i+j] in nums:
#
#                     element_quantity = element_quantity + string[i+j]
#
#                 else:
#                     on = False
#                     if i+j == (len(string)-1):
#                         lista.append((string[i+j],'1'))
#
#
#                     break
#                 if len(element_quantity)+len(element_name) == len(string):
#                     lista.append((element_name,element_quantity))
#                     return
#
#         i +=1
#     lista.append((element_name,element_quantity))
#
#     if i+j < len(string) and string[i+j-1] != nums :
#         string = string[i+j-1:]
#         from_string_to_dict(string,lista)
#
#
# def normalize_formula(lista):
#     """ Normalize quantity of chemical formula to 100 unit size cell """
#
#     dict_chemical = {symbol: float(quantity) for symbol, quantity in lista}
#     n_atoms = sum(list(dict_chemical.values()))
#     factor = 100.0/n_atoms
#
#     lista_normalized = [(symbol,str(round(factor*float(quantity),1))) for symbol, quantity in lista]
#
#     return lista_normalized
