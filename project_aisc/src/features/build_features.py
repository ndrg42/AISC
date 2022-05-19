import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as np_train_test_split
import sys
sys.path.append('../data')
sys.path.append('src/utils')
import utils


#Add input method in SuperConData
#add remove douplicated row in SuperConData
#Put in another file the function

class AtomData():
    """Class that holds the data of the periodic table and processes them.

        Categorical data is encoded with both natural number mapping and one hot encoding.
        Numerical data is impotated with the mean value and standardized.
        Lanthanides and antanides have group set equal to 0 before build_features

        Attributes:
            periodic_table: pandas DataFrame that holds non processed data
            atom_data: pandas DataFrame that holds processed data

        Methods:
            get_atom_data: return processed periodic table data
            bulid_atom_data: process and store periodic table data in atom_data
            get_numerical_data_processed: return a pd.DataFrame with numerical data processed
            get_categorical_data_processed: return a pd.DataFrame with categorical data processed
            one_hot_encode: one hot encode a feature avaible in self.periodic_table
            natural_encode: map feature's value into natural number

    """

    def __init__(self,periodic_table):
        """Inits AtomData with periodic table data"""

        self.periodic_table = periodic_table
        self.atom_data = None

    def one_hot_encode(self,atomic_feature):
        """Select atomic_feature in self.periodic_table and one hot encode it"""

        one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        #Nan value are set equals to Void (imputation) and encoded as a separate class
        uncoded_atomic_feature = np.array(self.periodic_table[atomic_feature].fillna(value='Void')).reshape(-1,1)
        one_hot_feature_encoded = pd.DataFrame(one_hot_encoder.fit_transform(uncoded_atomic_feature))

        # Put index back after one hot encode action (OneHotEncoder remove index)
        one_hot_feature_encoded.index = self.periodic_table.index

        return one_hot_feature_encoded


    def natural_encode(self,atomic_feature,atomic_map):
        """Map feature's value into natural number in according to atomic_map (dict)"""

        natural_feature_encoded = self.periodic_table[atomic_feature].map(atomic_map)

        return natural_feature_encoded



    def get_categorical_data_processed(self,categorical_encode_plan = {'lattice_structure':'one-hot','block':{ 's': 0,'p' : 1,'d' : 2,'f' : 3}}):
        """Return categorical data encoded"""

        categorical_dataset_list = []

        for categorical_feature in categorical_encode_plan.keys():
            if categorical_encode_plan[categorical_feature] == 'one-hot':
                categorical_dataset_list.append(self.one_hot_encode(categorical_feature))
            else:
                categorical_dataset_list.append(self.natural_encode(atomic_feature=categorical_feature,atomic_map=categorical_encode_plan[categorical_feature]))

        categorical_dataset = pd.concat(categorical_dataset_list, axis=1)

        return categorical_dataset


    def get_numerical_data_processed(self,numerical_columns):
        """Return a DataFrame with numerical data processed: Imputed by mean and standardinzed"""

        numeric_periodic_table = self.periodic_table[numerical_columns].copy()
        # Impute the missing value with the mean of the avaible data and standardize them
        my_imputer = SimpleImputer(strategy='mean')

        transformer = preprocessing.StandardScaler()
        imputed_periodic_table = pd.DataFrame(my_imputer.fit_transform(numeric_periodic_table))
        imputed_standardized_periodic_table = pd.DataFrame(transformer.fit_transform(imputed_periodic_table))

        # Imputation removed column names so we put them back
        imputed_standardized_periodic_table.columns = numeric_periodic_table.columns

        return imputed_standardized_periodic_table




    def build_atom_data(self):
        """
        Fill self.atom_data with periodic table data imputed, encoded and processed.
        """

        # Select numerical columns for imputations and processation
        numerical_columns = [feature for feature in self.periodic_table.columns if
                      self.periodic_table[feature].dtype in ['int64', 'float64','int32', 'float32']]

        # Lanthanides and antanides don't have group.
        # We choose to set them to 0 before build_features (it's an unique value for them)

        if 'group_id' in numerical_columns:
            self.periodic_table.loc[:,'group_id'] = self.periodic_table['group_id'].fillna(0)


        categorical_data = self.get_categorical_data_processed(categorical_encode_plan = {'lattice_structure':'one-hot','block':{ 's': 0,'p' : 1,'d' : 2,'f' : 3}})
        numerical_data = self.get_numerical_data_processed(numerical_columns)


        self.atom_data = pd.concat([numerical_data,categorical_data], axis= 1)

    def get_atom_data(self):
        """Return periodic table data processed"""

        #if periodic table data is not processed, we process it and then return it
        if self.atom_data is None:
            self.build_atom_data()

        return self.atom_data



class SuperConData():
    """Class processes SuperCon dataset to make it ready as model input.

        The class holds the relevant data to build the input for the machine learning
        model. It can build vector representation of the material, i.e. assemble
        features computed by function from atomic features (analitic dataset).
        In this case the input is an array.
        Or it can build list of atomic representation: in this case the input is a list
        and each element in the list is an array representing the chemical element.

        Attributes:
            atom_data: pandas DataFrame that holds processed data of atoms
            supercon_dataset: pandas DataFrame that holds SuperCon data
            dataset: dataset ready to be processed by the model
            padding: (default auto) pads input if needed. Set it manual or let it infer the value

        Methods:
            get_dataset: return processed periodic table data
            bulid_dataset: assemble and store dataset ready for model input
            select_atom_in_formula: select the atoms and relative quantity written in the chemical formula
            get_atom_arrays: return arrays filled with atom data and padding
            expand_row_into_model_input: turn a row into a model input


    """

    def __init__(self,atom_data,supercon_dataset,padding = 'auto'):
        """Inits SuperConData. Processed atoms data and SuperCon are mandatory"""
        self.atom_data = atom_data
        self.supercon_dataset = supercon_dataset
        self.supercon_dataset_processed = None
        self.analytical_supercon_dataset_processed = None
        self._padding = padding

    def get_dataset(self):
        """Return dataset ready for model"""

        if self.supercon_dataset_processed is None:
            self.build_dataset()

        return self.supercon_dataset_processed


    def build_dataset(self):
        """Interate over the rows, build model input and store it into dataset attributes"""

        supercon_dataset = self.supercon_dataset.copy()
        #We keep only chemical symbol columns(are 96)
        supercon_dataset = supercon_dataset.iloc[:,:96]
        #Map atom symbol into natural number:  symbol-> Z-1 where Z: atomic number
        supercon_dataset.columns = range(96)

        max_elements_in_formula = (supercon_dataset>0).sum(axis =1).max()
        #Set padding if needed for input generation
        if self._padding == 'auto':
            self._padding = max_elements_in_formula

        dataset_ready_for_model = supercon_dataset.apply(self.expand_row_into_model_input,axis=1)
        #Rearrange type and dimension order to fit model input
        #It's a list of #samples and each samples has associated an array
        # list (np.array.shape = (#samples,atom_representation) )

        dataset_ready_for_model = np.moveaxis(np.array(list(dataset_ready_for_model)),0,1)
        self.supercon_dataset_processed = list(dataset_ready_for_model)

    def select_atom_in_formula(self,formula_array):
        """Return a list of tuples with atomic number shifted by 1(Z-1) and atomic quantity"""

        #remove chemical elements not present in the chemical formula
        formula_array_other_than_0 = formula_array[formula_array>0]
        #atoms_index is the atomic number (Z) -1
        atoms_index = formula_array_other_than_0.index
        #quantity of the relative element in the formula
        atoms_value = formula_array_other_than_0.values

        atom_symbol_quantity = [(i,j) for i,j in zip(atoms_index,atoms_value)]

        return atom_symbol_quantity

    def get_atom_arrays(self,atom_symbol_quantity):
        """Return an array(numpy) of lenght max_length filled by atomic arrays and 0's arrays (padding)"""

        list_atom_features = []
        #The symbol is not a string (like 'H') but an index (like 0 for 'H')
        #symbol = Z-1 where Z = atomic number
        for symbol,quantity in atom_symbol_quantity:
            atom_features = self.atom_data.iloc[symbol,:].to_numpy()
            complete_atom_features = np.append(atom_features,quantity)
            list_atom_features.append(complete_atom_features)

        padding_value = self._padding - len(atom_symbol_quantity)
        #Padding need to be > 0. It is ensured b contruction normally
        assert padding_value >= 0,f'padding_value: {padding_value} and atoms in formula: {atom_symbol_quantity}'

        array_atom_features_padded = np.pad(list_atom_features,[(0,padding_value),(0,0)],)

        return array_atom_features_padded

    def expand_row_into_model_input(self,row):
        """Expand a row (pandas Series) into a model ready input"""
        #Select atom with quantity different from 0
        #And put them into a list of tuples (index_symbol,quantity)
        atom_symbol_quantity = self.select_atom_in_formula(row)

        expanded_row = self.get_atom_arrays(atom_symbol_quantity)

        return expanded_row

    def get_analytical_dataset(self):
        """Return dataset ready for model"""

        if self.analytical_supercon_dataset_processed is None:
            self.build_analytical_dataset()

        return self.analytical_supercon_dataset_processed

    def build_analytical_dataset(self):

        supercon_dataset = self.supercon_dataset.copy()
        #We keep only chemical symbol columns(are 96)
        supercon_dataset = supercon_dataset.iloc[:,:96]
        #Map atom symbol into natural number:  symbol-> Z-1 where Z: atomic number
        supercon_dataset.columns = range(96)
        string_mono_functions = ['mean','geometric_average','entropy','range_feature','std']
        string_bi_functions = ['weighted_mean','weighted_geo_mean','weighted_entropy','weighted_range_feature','weighted_std']
        selected_features = ['atomic_weight','ionenergies','valence','thermal_conductivity','atomic_radius_rahm','density','electron_affinity','fusion_heat']

        dataset_columns = [
            *[[func+'_'+ feature for func in string_mono_functions] for feature in selected_features],
            *[[func +'_'+ feature for func in string_bi_functions] for feature in selected_features]
            ]
        dataset_columns = [feature for sublist in dataset_columns for feature in sublist]

        dataset_ready_for_model = supercon_dataset.apply(self.expand_row_into_analytical_model_input,axis=1)
        self.analytical_supercon_dataset_processed = pd.DataFrame(list(dataset_ready_for_model))
        self.analytical_supercon_dataset_processed.columns = dataset_columns

    def expand_row_into_analytical_model_input(self,row):
        """Expand a row (pandas Series) into an analitical model ready input with lenght 80"""
        #Select atom with quantity different from 0
        #And put them into a list of tuples (index_symbol,quantity)
        atom_symbol_quantity = self.select_atom_in_formula(row)
        expanded_row = []
        selected_features = ['atomic_weight','ionenergies','valence','thermal_conductivity','atomic_radius_rahm','density','electron_affinity','fusion_heat']
        atoms_quantity = [quantity for _,quantity in atom_symbol_quantity]
        atoms_symbol = [symbol for symbol,_ in atom_symbol_quantity]
        selected_atom = self.atom_data.loc[atoms_symbol,selected_features]

        analitic_mono_functions = [np.mean, utils.geo_mean, utils.entropy, utils.range_feature, np.std,]
        for foo in analitic_mono_functions:
            expanded_row.append(selected_atom.apply(foo).values)

        analitic_bi_functions = [utils.weighted_average, utils.weighted_geo_mean, utils.weighted_entropy, utils.weighted_range_feature, utils.weighted_std]
        for foo in analitic_bi_functions:
            expanded_row.append(selected_atom.apply(lambda x: foo(x,atoms_quantity),axis=0).values)

        expanded_row = np.reshape(np.array(expanded_row),(-1,))
        return expanded_row

    # def weighted_average(self,med,perc):
    #     return np.average(med,weights=perc)
    #
    # def geo_mean(self,iterable):
    #     a = np.abs(iterable)
    #     return a.prod()**(1.0/len(a))
    #
    # def weighted_geo_mean(self,med,perc):
    #     a = np.abs(med)**(perc/np.sum(perc))
    #     return a.prod()**(1.0/len(a))
    #
    #
    #
    # def entropy(self,med):
    #     med = np.abs(med)
    #     med = np.where(med>0.00000000001,med,0.00000000001)
    #     return -np.sum(med*np.log(med))
    #
    #
    # def weighted_entropy(self,med,perc):
    #     med = np.abs(med)
    #     med = np.where(med>0.00000000001,med,0.00000000001)
    #     med = med*perc/np.sum(med*perc)
    #
    #     return -np.sum(med*np.log(med))
    #
    #
    # def range_feature(self,med):
    #     max = med.max()
    #     min = med.min()
    #     return max-min
    #
    # def weighted_range_feature(self,med,perc):
    #     med = med*perc/np.sum(perc)
    #
    #     return self.range_feature(med)
    #
    # def weighted_std(self,med,perc):
    #     x = abs(med - np.average(med,weights = perc))**2
    #     std = np.sqrt(np.average(x,weights = perc))
    #     return std


def train_test_split(data,label,test_size=0.2):
    """Custom train-test split.

       Args:
           data: A list of numpy array containing atom's representation
           label: List or numpy array or pandas Series containing labels
       Returns:
           X,X_test,y,y_test: A tuple containing in order data, test's data, label, test's label
       """

    X,X_test,y,y_test = np_train_test_split(np.moveaxis(np.array(data),0,1),label,test_size = test_size)
    X  = list(np.moveaxis(X,0,1))
    X_test  = list(np.moveaxis(X_test,0,1))

    return X,X_test,y,y_test








class DataProcessor():

    def __init__(self,ptable,sc_dataframe):
        self.ptable = ptable
        self.sc_dataframe = sc_dataframe
        self.Atom = None
        self.dataset = None
        self.t_c = sc_dataframe['critical_temp']
        self.max_lunghezza = 10
        self.analitic_dataset = None

    def get_input(self,compound):
        from mendeleev import element
        from chela.formula_handler import from_string_to_dict

        d= []
        from_string_to_dict(compound,d)

        input_dim = 33
        nulla = np.zeros(input_dim)
        entrata = []

        try:
            if self.Atom == None:
                super().build_Atom()
        except:
            pass

        for j in range(self.max_lunghezza):
            if j < len(d):
                entrata.append(np.array(self.Atom.loc[element(d[j][0]).atomic_number -1].append(pd.Series(float(d[j][1])))))
            else:
                entrata.append(np.array(pd.Series(nulla)))

        e = np.array(entrata)
        e = list(np.expand_dims(e,axis = 1))

        return e



    def input_from_dict(self,d_m):
        from mendeleev import element

        input_dim = 33

        try:
            if self.Atom == None:
                super().build_Atom()
        except:
            pass

        nulla = np.zeros(input_dim)
        entrata = []
        for j in range(self.max_lunghezza):
            if j < len(d_m):
                entrata.append(np.array(self.Atom.loc[(element(list(d_m.keys())[j]).atomic_number -1)].append(pd.Series(list(d_m.values())[j]))))
            else:
                entrata.append(np.array(pd.Series(nulla)))

        e = np.array(entrata)
        e = list(np.expand_dims(e,axis = 1))

        return e



    def build_Atom(self):
        """
        dataset contenente i dati della tavola periodica processati, ovvero normalizzati ed Imputation
        i dati numeri sono normalizzati rispetto al valore massimo ed imputati con la media
        i dati categorici sono: la struttura cristallina, codificata tramite one hot encoding, ed il guscio (s,p,d,f) codificati tramite numeri crescenti

        input:
             PeriodicTable

        output:
             Atom
        """

        #Le feature a livello atomico utilizzate sono solo 8:
        #Atomic Mass, First Ionization Energy, Atomic Radius,
        #Density, Electron Affinity, Fusion Heat, Thermal Conductivity, Valence.
        feature_chosen = ['atomic_weight','ionenergies','valence','thermal_conductivity',
                           'atomic_radius_rahm','density','electron_affinity','fusion_heat']


        categorical_cols = [cname for cname in self.ptable.columns if
                            self.ptable[cname].dtype == "object"]


        # Select numerical columns
        numerical_cols = [cname for cname in self.ptable.columns if
                      self.ptable[cname].dtype in ['int64', 'float64','int32', 'float32']]


        X_train_num = self.ptable[numerical_cols].copy()
        # Imputation
        my_imputer = SimpleImputer(strategy='mean')
        # transformer = prebuild_features.Normalizer(norm = 'max')
        transformer = preprocessing.StandardScaler()
        imputed_X_train_num = pd.DataFrame(my_imputer.fit_transform(X_train_num))
        imputed_X_train_num = pd.DataFrame(transformer.fit_transform(imputed_X_train_num))



        # Imputation removed column names; put them back
        imputed_X_train_num.columns = X_train_num.columns

        X_train = imputed_X_train_num

        cols_with_missing = [col for col in X_train.columns
                             if X_train[col].isnull().any()]


        #Apply one-hot encoder to each column with categorical data


        OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        lattice_encoded = np.array(self.ptable['lattice_structure'].fillna(value='Void')).reshape(-1,1)
        OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(lattice_encoded))

        # One-hot encoding removed index; put it back
        OH_cols_train.index = X_train.index

        # Add one-hot encoded columns to numerical features
        X_train = pd.concat([X_train, OH_cols_train], axis=1)


        labelencoder = LabelEncoder()
        Label_cols_train = self.ptable['block'].map({ 's': 0,
                                                 'p' : 1,
                                                 'd' : 2,
                                                 'f' : 3
                                                 })

        self.Atom = pd.concat([X_train, Label_cols_train], axis= 1)


    def build_analitic_dataset(self):
        """
        dataset contenente i dati della tavola periodica processati, ovvero normalizzati ed Imputation
        i dati numeri sono normalizzati rispetto al valore massimo ed imputati con la media
        i dati categorici sono: la struttura cristallina, codificata tramite one hot encoding, ed il guscio (s,p,d,f) codificati tramite numeri crescenti

        input:
             PeriodicTable

        output:
             analitic dataset
        """

        #Le feature a livello atomico utilizzate sono solo 8:
        #Atomic Mass, First Ionization Energy, Atomic Radius,
        #Density, Electron Affinity, Fusion Heat, Thermal Conductivity, Valence.
        feature_chosen = ['atomic_weight','ionenergies','valence','thermal_conductivity',
                           'atomic_radius_rahm','density','electron_affinity','fusion_heat']
        #a media, media pesata, media geometrica, media geometrica pesata,
        # entropia, entropia pesata,
        #range, range pesato, deviazione standard, deviazione standard pesata.


        n_comp = len(self.t_c)
        lista_atomi = self.sc_dataframe[0:n_comp]


        entrata = []
        sin_comp = []
        trans_comp = []
        Lista_atomi = lista_atomi.to_numpy()
        for j in range(lista_atomi.shape[0]):
            for i in range(lista_atomi.shape[1]-2):
                if Lista_atomi[j][i] >0:
                    trans_comp.append({ 'atom' :i , 'perc(%)': Lista_atomi[j][i]})
            sin_comp.append(trans_comp)
            trans_comp = []



        med = np.array([])
        perc = np.array([])
        composto = []
        C = []
        for k in range(len(sin_comp)):
            composto = []
            for s in feature_chosen:
                med = np.array([])
                perc = np.array([])
                for i in range(len(sin_comp[k])):
                    med = np.append(med,self.Atom[s].iloc[sin_comp[k][i]['atom']])
                    perc = np.append(perc,sin_comp[k][i]['perc(%)'])
                composto.append([med.mean(),np.average(med,weights=perc),
                self.geo_mean(med),self.weighted_geo_mean(med,perc),self.entropy(med),self.weighted_entropy(med,perc),
                self.range_feature(med),self.weighted_range_feature(med,perc),np.std(med),self.weighted_std(med,perc)])
            if k == 0:
                C = np.array(composto).reshape(1,80)
            else:
                C = np.concatenate((C,np.array(composto).reshape(1,80)))


        self.analitic_dataset = pd.DataFrame(C)



    def build_dataset(self):
        """
        dataset contenente i dati della tavola periodica processati, ovvero normalizzati ed Imputation
        i dati numeri sono normalizzati rispetto al valore massimo ed imputati con la media
        i dati categorici sono: la struttura cristallina, codificata tramite one hot encoding, ed il guscio (s,p,d,f) codificati tramite numeri crescenti

        input:
             PeriodicTable

        output:
              dataset
        """

        n_comp = len(self.t_c)
        lista_atomi = self.sc_dataframe[0:n_comp]


        entrata = []
        sin_comp = []
        trans_comp = []
        Lista_atomi = lista_atomi.to_numpy()
        for j in range(lista_atomi.shape[0]):
            for i in range(lista_atomi.shape[1]-2):
                if Lista_atomi[j][i] >0:
                    trans_comp.append({ 'atom' :i , 'perc(%)': Lista_atomi[j][i]})
            sin_comp.append(trans_comp)
            trans_comp = []

        input_dim = self.Atom.shape[1]+1

        lunghezza_dei_composti = [len(sin_comp[x]) for x in range(len(sin_comp))]
#        self.max_lunghezza = max(lunghezza_dei_composti)
        max_lunghezza  = self.max_lunghezza

        nulla = np.zeros(input_dim)
        count = 0
        entrata = []
        for i in range(len(sin_comp)):
            for j in range(max_lunghezza):
                if j < len(sin_comp[i]):
                    try:
                        entrata.append(self.Atom.loc[sin_comp[i][j]['atom']].append(pd.Series(sin_comp[i][j]['perc(%)'])))
                    except:
                        print(sin_comp[i][j]['atom'])
                else:
                    entrata.append(pd.Series(nulla))



        entrata_list_numpy = [np.array([entrata[x+y] for y in range(max_lunghezza)]) for x in range(0,len(entrata),max_lunghezza)]
        self.dataset = list(np.moveaxis(np.array(entrata_list_numpy),0,1))


    def save_data_csv(self,path,name,tc= False):

        dataset = self.dataset

        if tc == False:
            X = np.asarray(dataset)
        else:
            self.t_c.to_csv(path +name,index= False)

        try:
            m,n,r = X.shape
            X = np.reshape(X,(n,m*r))
            X = pd.DataFrame(X)
            X.to_csv(path + name,index = False)
        except:
            pass



    def load_data_processed(self,path):

        X = pd.read_csv(path)
        if X.shape[1] == 1:
            X = np.ravel(np.asarray(X))
            self.t_c = X
        else:
            X = np.reshape(np.asarray(X),(10,np.asarray(X).shape[0],33))
            self.dataset = list(X)

    def hist_t_c(self,fit = False,smooth = False):

        if fit == False:
            sns.histplot(t_c)
        else:
            if smooth == False:
                sns.histplot(t_c,kde = True)
            else:
                sns.kdeplot(t_c,shade = True)

    def train_test_split(self,data,t_c,test_size=0.2):

        X,X_test,y,y_test = train_test_split(np.moveaxis(np.array(data),0,1),np.array(t_c),test_size = test_size)
        X  = list(np.moveaxis(X,0,1))
        X_test  = list(np.moveaxis(X_test,0,1))

        return X,X_test,y,y_test

    def geo_mean(self,iterable):
        a = np.abs(iterable)
        return a.prod()**(1.0/len(a))

    def weighted_geo_mean(self,med,perc):
        a = np.abs(med)**(perc/np.sum(perc))
        return a.prod()**(1.0/len(a))



    def entropy(self,med):
        med = np.abs(med)
        med = np.where(med>0.00000000001,med,0.00000000001)
        return -np.sum(med*np.log(med))


    def weighted_entropy(self,med,perc):
        med = np.abs(med)
        med = np.where(med>0.00000000001,med,0.00000000001)
        med = med*perc/np.sum(med*perc)

        return -np.sum(med*np.log(med))


    def range_feature(self,med):
        max = med.max()
        min = med.min()
        return max-min

    def weighted_range_feature(self,med,perc):
        med = med*perc/np.sum(perc)

        return self.range_feature(med)

    def weighted_std(self,med,perc):
        x = abs(med - np.average(med,weights = perc))**2
        std = np.sqrt(np.average(x,weights = perc))
        return std


def remove_columns_with_missing_elements(dataset, max_missing_value = 30,exceptions = None):
    """remove columns that has more than max_missing_value with expection for except columns"""

    empty_columns = [column for column in dataset.columns if dataset[column].isna().sum()> max_missing_value]

    #remove from the list exceptional columns even if they have too many missing values
    if exceptions:
        for column in exceptions:
            if column in empty_columns:
                empty_columns.remove(column)

    return dataset.drop(columns = empty_columns)
