import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../data')
import DataLoader

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

        num = ['0','1','2','3','4','5','6','7','8','9','.']
        key=''
        d= {}
        value = ''
        ok = False
        compound = compound
        for s in compound:
            if s not in num:
                if ok == True:
                    d.update({key:float(value)})
                    key = ''
                    value = ''
                    ok = False

                key = key+s
            if s in num:
                value = value +s
                ok = True
        if ok == False:
            d.update({key:1.0})
        if ok == True:
            d.update({key:float(value)})

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
                entrata.append(np.array(self.Atom.loc[(element(list(d.keys())[j]).atomic_number -1)].append(pd.Series(list(d.values())[j]))))
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
                      self.ptable[cname].dtype in ['int64', 'float64']]


        X_train_num = self.ptable[numerical_cols].copy()
        # Imputation
        my_imputer = SimpleImputer(strategy='mean')
        transformer = preprocessing.Normalizer(norm = 'max')
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
                for i in range(len(sin_comp[0])):
                    med = np.append(med,self.Atom[s].iloc[sin_comp[0][i]['atom']])
                    perc = np.append(perc,sin_comp[0][i]['perc(%)'])
                composto.append([med.mean(),np.average(med,weights=perc),
                geo_mean(med),weighted_geo_mean(med,perc),entropy(med),weighted_entropy(med,perc),
                range_feature(med),weighted_range_feature(med,perc),np.std(med),weighted_std(med,perc)])
            if k == 0:
                C = np.array(composto).reshape(80,1)
            else:
                C = np.hstack((C,np.array(composto).reshape(80,1)))

        C = np.moveaxis(C,0,1)

        transformer = preprocessing.Normalizer(norm = 'max')

        self.analitic_dataset = pd.DataFrame(transformer.fit_transform(C))



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
            self.t_c.to_csv(path +'dataset_label_elaborated.csv',index= False)

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
        a = np.array(iterable)
        return a.prod()**(1.0/len(a))

    def weighted_geo_mean(self,med,perc):
        a = med**(perc/np.sum(perc))
        return a.prod()**(1.0/len(a))



    def entropy(self,med):
        return -np.sum(med*np.log(med))


    def weighted_entropy(self,med,perc):
        med = med*perc/np.sum(med*perc)

        return -np.sum(med*np.log(med))


    def range_feature(self,med):
        max = med.max()
        min = med.min()
        return max-min

    def weighted_range_feature(self,med,perc):
        med = med*perc/np.sum(perc)

        return range_feature(med)

    def weighted_std(self,med,perc):
        x = abs(med - np.average(med,weights = perc))**2
        std = np.sqrt(np.average(x,weights = perc))
        return std
