
import pandas as pd

def SuperCon(sc_path = '../data/raw/unique_m.csv'):
    """
    carica un dataset con gli atomi che compongono i materiali superconduttori e la loro temperatura critica

    output:
          dataset
    """

    sc_dataframe = pd.read_csv(sc_path)
    return sc_dataframe


def PeriodicTable(max_index_atom=109,max_missing_value=30):
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
    from mendeleev import get_table
    from mendeleev import element

    periodic_table = get_table('elements')
    #max_index_atom = 109
    #max_missing_value = 30
    #PROVA: TOLGO "en_pauling","group_id","evaporation_heat"
    prop_atomic_unn = ['annotation','description','name','jmol_color','symbol','is_radioactive','vdw_radius_mm3',
                           'cpk_color','uses','sources','name_origin','discovery_location','covalent_radius_cordero',
                           'discoverers','cas','goldschmidt_class','molcas_gv_color','discovery_year','atomic_radius','series_id',
                           'electronic_configuration','glawe_number','en_ghosh','heat_of_formation','covalent_radius_pyykko_double',
                           'vdw_radius_alvarez','abundance_crust', 'abundance_sea', 'c6_gb','vdw_radius_uff',
                           'dipole_polarizability_unc','boiling_point','pettifor_number','mendeleev_number']

    ionenergies_col= []
#non è disponibile il dato per i maggiore di 109
    for i in range(1,max_index_atom):
        el = element(i)
        el = el.ionenergies[1]
        ionenergies_col.append(el)

    valence_col = []
    for i in range(1,max_index_atom):
        el = element(i)
        el = el.nvalence()
        valence_col.append(el)

    periodic_table.drop(prop_atomic_unn,axis = 1,inplace=True)
    periodic_table = periodic_table[:(max_index_atom-1)]

    periodic_table['valence'] = valence_col
    periodic_table['ionenergies'] = ionenergies_col


    periodic_table.shape
    col_vuote = []
    mancanti = periodic_table.isna().sum()
    for i in range(mancanti.size):
        if mancanti[i] >= max_missing_value:
            col_vuote.append(mancanti.index[i])


    col_vuote.remove('thermal_conductivity')
    col_vuote.remove('fusion_heat')
    col_vuote.remove('electron_affinity')
    periodic_table.drop(col_vuote,axis = 1,inplace=True)
    periodic_table = periodic_table[:96]

    import numpy as np

    path = "/home/claudio/aisc/project_aisc/data/raw/"
    periodic_table = pd.DataFrame(periodic_table)
    thermal_conductivity = pd.read_csv(path+"thermal_conductivity.csv",header=None)

    thermal_conductivity.replace('QuantityMagnitude[Missing["NotAvailable"]]',np.nan,inplace=True)
    thermal_conductivity.replace('QuantityMagnitude[Missing["Unknown"]]',np.nan,inplace=True)

    specific_heat = pd.read_csv(path+"specific_heat.csv",header = None)

    specific_heat.replace('QuantityMagnitude[Missing["NotAvailable"]]',np.nan,inplace=True)
    specific_heat.replace('QuantityMagnitude[Missing["Unknown"]]',np.nan,inplace=True)

    electron_affinity = pd.read_csv(path+"electron_affinity.csv",header=None)

    electron_affinity.replace('QuantityMagnitude[Missing["NotAvailable"]]',np.nan,inplace=True)
    electron_affinity.replace('QuantityMagnitude[Missing["Unknown"]]',np.nan,inplace=True)

    density = pd.read_csv(path+"density.csv",header=None)

    density.replace('QuantityMagnitude[Missing["NotAvailable"]]',np.nan,inplace=True)
    density.replace('QuantityMagnitude[Missing["Unknown"]]',np.nan,inplace=True)

    for i in range(96):

        if periodic_table["thermal_conductivity"].isna()[i]:
           periodic_table["thermal_conductivity"][i] =thermal_conductivity.values[i]

        if periodic_table["specific_heat"].isna()[i]:
            periodic_table["specific_heat"][i] = specific_heat.astype('float32').values[i]/1000

        if periodic_table["electron_affinity"].isna()[i]:
            periodic_table["electron_affinity"][i] = electron_affinity.astype('float32').values[i]/100

        if periodic_table["density"].isna()[i]:
            periodic_table["density"][i] = density.astype('float32').values[i]/1000


    return periodic_table



#%%
import pandas as pd
supercon = pd.read_csv('../../data/raw/SuperCon_database.dat',delimiter = r"\s+",names = ['formula','tc'])
#conto il numero di nan
supercon['tc'].isna().replace(False,0).sum()
#remove rows with nan value on tc
supercon = supercon.dropna()
#check duplicate row
duplicated_row = supercon[supercon.duplicated(subset = ['formula'],keep = False)].groupby(supercon['formula']).aggregate({'tc':'mean'}).reset_index()
duplicated_row

supercon[supercon['tc'].isnull()]
supercon.drop_duplicates(subset = ['formula'],inplace = True,keep = False)
supercon[supercon.duplicated(subset = ['formula'],keep = False)]
supercon[supercon['formula']=='ZrS3']

supercon.duplicated(subset = ['formula'],keep = False).replace(True,1).sum()
supercon.shape
supercon.head()
supercon[supercon['tc']==0].shape
supercon= supercon.append(duplicated_row,ignore_index=True)


from mendeleev import element
list_element = []
element(2).symbol
index = []
for i in supercon['formula'].index:
    mol = supercon['formula'][i]
    if 'x' in mol:
        index.append(i)

sc_dict={}
num_element = 96
for i in range(1,num_element+1):
    sc_dict[element(i).symbol] = []

sc_dict['material'] = []
sc_dict['critical_temp'] = []
sc_dict
list_element = list(sc_dict.keys())[:num_element]

nums = ['0','1','2','3','4','5','6','7','8','9','.']
indaga = []
l = []
stringa = 'YSe2Se4'
from_string_to_dict(stringa,l)
l
supercon['formula'][0]
for i in range(supercon['formula'].shape[0]):

    sc_string = supercon['formula'][i]
    sc_dict['material'].append(sc_string)
    sc_dict['critical_temp'].append(float(supercon['tc'][i]))
    tupl_atom = []
    from_string_to_dict(sc_string,tupl_atom)
    list_atom = []
    for j in range(len(tupl_atom)):
        list_atom.append(tupl_atom[j][0])

        if tupl_atom[j][0] in list_element:
            sc_dict[tupl_atom[j][0]].append(float(tupl_atom[j][1]))

    pozzo = element_list - (set(list_atom))
    if len(list(set(list_atom))) != len(list_atom):
        occhio = list_atom
        indaga.append(i)
    for o in pozzo:
        sc_dict[o].append(0)

supercon.drop(indaga,inplace=True)
supercon.reset_index(inplace=True)
len(indaga)
occhio
sc_dict
store = []
for k in list(sc_dict.keys()):
    if len(sc_dict[k]) != 1000:
        print(k)
    store.append(len(sc_dict[k]))

store
list_atom
len(element_list.difference(set(list_atom)))
element_list = set(element_list)
element_list = list(sc_dict.keys())
element_list = element_list[:-2]
len(element_list)
for j in range(len(tupl_atom)):
    if tupl_atom[j][0] in list_element:
        sc_dict[tupl_atom[j][0]] =sc_dict[tupl_atom[j][0]].append(float(tupl_atom[j][1]))

sc_dict['material'] = []
sc_dict['material'].append(sc_string)
pd.DataFrame(sc_dict)
#%%

def from_string_to_dict(string,lista):
    nums = ['0','1','2','3','4','5','6','7','8','9','.']
    i = 0
    element_name = ''
    element_quantity = ''
    on = True


    while(i<len(string) and on ):
        if string[i] not in nums:

            element_name = element_name + string[i]

            if i == len(string)-1:
                lista.append((element_name,'1'))
                return
            if i+1 < len(string):
                if string[i+1].isupper() :
                    lista.append((element_name,'1'))
                    string = string[i+1:]
                    from_string_to_dict(string,lista)
                    return

            if i == len(stringa)-1:
                lista.append((element_name,'1'))
                return

        if string[i] in nums:
            element_quantity = ''
            for j in range(len(string)-i):

                if string[i+j] in nums:

                    element_quantity = element_quantity + string[i+j]

                else:
                    on = False


                    break
                if i+j+len(element_name)-1 == len(string):
                    lista.append((element_name,element_quantity))
                    return

        i +=1
    lista.append((element_name,element_quantity))

    if i+j < len(string) and string[i+j-1] != nums :
        string = string[i+j-1:]
        from_string_to_dict(string,lista)






#%%
sc = pd.read_csv('../../data/raw/unique_m.csv')
sc.shape
sc['material'].duplicated(keep = False).replace(True,1).sum()
duplicated_row = sc[sc.duplicated(subset = ['material'],keep = False)].groupby(sc['material']).aggregate({'critical_temp':'mean'}).reset_index()
sc.drop_duplicates(subset = ['material'],inplace = True,keep = False)
sc = sc.append(duplicated_row,ignore_index=True)
dup = []
index= sc[sc.duplicated(subset = 'material',keep = False)]['material'].index
for i in index:
    for j in index[i+1:]:
        if sc['material'][i] == sc['material'][j]:
            dup.append((i,j))

len(dup)
sc['material'][55]
sc['material'][328]
sc['material'][375]
sc['material'][376]
sc['material'][469]
sc['material'][576]
duplicated_row.duplicated().replace(True,1).sum()
