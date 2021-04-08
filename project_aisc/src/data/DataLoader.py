
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
                           'dipole_polarizability_unc','boiling_point','pettifor_number','mendeleev_number',"en_pauling","group_id","evaporation_heat"]

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
