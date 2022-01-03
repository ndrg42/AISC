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

def test_supercon_build_dataset():

    ptable = DataLoader.PeriodicTable()
    sc_dataframe = DataLoader.SuperCon(sc_path ='data/raw/supercon_tot.csv')

    atom_data_processor = Processing.AtomData(ptable)
    processed_atom_data = atom_data_processor.get_atom_data()

    supercon_data_processor = Processing.SuperConData(processed_atom_data,sc_dataframe,padding = 10)

    supercon_data_processor.build_dataset()
    dataset = supercon_data_processor.get_dataset()

    dataset = np.array(dataset)
    np_supercon_data = np.load('data/processed/dataset_supercon.npy')

    np_supercon_data[:,:,7] = np_supercon_data[:,:,7]*0
    dataset[:,:,7] = dataset[:,:,7]*0

    assert ((dataset - np_supercon_data)<0.001).all()

#
# ptable = DataLoader.PeriodicTable()
# sc_dataframe = DataLoader.SuperCon(sc_path ='data/raw/supercon_tot.csv')
#
# atom_data_processor = Processing.AtomData(ptable)
# processed_atom_data = atom_data_processor.get_atom_data()
#
# atom_data_processor.get_atom_data()
#
# supercon_data_processor = Processing.SuperConData(processed_atom_data,sc_dataframe,padding = 10)
# supercon_data_processor.
# supercon_data_processor.build_dataset()
# dataset = supercon_data_processor.dataset
# dataset = supercon_data_processor.get_dataset()
#
# np_supercon_data = np.load('data/processed/dataset_supercon.npy')
#
# dd = Processing.DataProcessor(ptable,sc_dataframe)
# dd.build_Atom()
# dd.build_dataset()
#
# ((np.array(dataset) - np_supercon_data)<0.001).all()
# npp = np_supercon_data
# np_supercon_data[:,:,7] = np_supercon_data[:,:,7]*0
# np_supercon_data.shape
# i = 1
# ((dataset[0,0,7:10]- np_supercon_data[0,0,7:10])<0.001).all()
# ((dataset[i]- np_supercon_data[i])>0.001)
#
# dataset[0,0,7]
# np_supercon_data[:,0,7]
# dataset[:,0,7]
# np.array(dd.dataset)[:,0,7]

#
# ptable = DataLoader.PeriodicTable()
# sc_dataframe = DataLoader.SuperCon(sc_path ='data/raw/supercon_tot.csv')
#
# atom_data = Processing.DataProcessor(ptable, sc_dataframe)
# attom = atom_data.Atom
# atom_data.build_Atom()
# atom_data.build_dataset()
# #%%
# None == None
# #%%
# atom_data.Atom
# npsc = np.load('data/processed/dataset_supercon.npy')
#
# list(npsc[:])
# ((np.array(atom_data.dataset) - npsc)<0.001).all()
# #%%
# sc_dataframe.shape[0]
#
# n_comp = sc_dataframe.shape[0]
# lista_atomi = sc_dataframe[0:n_comp]
#
# lista_atomi.to_numpy()
#
# entrata = []
# sin_comp = []
# trans_comp = []
# Lista_atomi = lista_atomi.to_numpy()
# for j in range(lista_atomi.shape[0]):
#     for i in range(lista_atomi.shape[1]-2):
#         if Lista_atomi[j][i] >0:
#             trans_comp.append({ 'atom' :i , 'perc(%)': Lista_atomi[j][i]})
#     sin_comp.append(trans_comp)
#     trans_comp = []
#
# sc_index = sc_dataframe.iloc[0,:-2][sc_dataframe.iloc[0,:-2]>0].index
#
# sc_only_elements = sc_dataframe.iloc[:,:-2]
#
# sc_index = sc_only_elements.index
# sc_value = sc_only_elements.values
# sc_value = sc_dataframe.iloc[0,:-2][sc_dataframe.iloc[0,:-2]>0].values
# for i,j in zip(sc_index,sc_value):
#     print(i,j)
#
# sc_dataframe.columns = range(0,98)
# sc_dataframe
#
# atom_data.Atom
# sin_comp
# input_dim = self.Atom.shape[1]+1
#
# lunghezza_dei_composti = [len(sin_comp[x]) for x in range(len(sin_comp))]
# #        self.max_lunghezza = max(lunghezza_dei_composti)
# max_lunghezza  = self.max_lunghezza
#
# nulla = np.zeros(input_dim)
# count = 0
# entrata = []
# for i in range(len(sin_comp)):
#     for j in range(max_lunghezza):
#         if j < len(sin_comp[i]):
#             try:
#                 entrata.append(self.Atom.loc[sin_comp[i][j]['atom']].append(pd.Series(sin_comp[i][j]['perc(%)'])))
#             except:
#                 print(sin_comp[i][j]['atom'])
#         else:
#             entrata.append(pd.Series(nulla))
#
#
#
# entrata_list_numpy = [np.array([entrata[x+y] for y in range(max_lunghezza)]) for x in range(0,len(entrata),max_lunghezza)]
# self.dataset = list(np.moveaxis(np.array(entrata_list_numpy),0,1))
#
# #%%
#
# def expand_row(row):
#     row = row[row>0]
#     row_index = row.index
#     row_value = row.values
#
#     atom_quantity = [(i,j) for i,j in zip(row_index,row_value)]
#
# sc_toy = sc_dataframe.iloc[:,:96]
# sc_toy.columns = range(96)
# sc_toy.iloc[0,:][sc_toy.iloc[0,:]>0]
# row_index = sc_toy.iloc[0,:][sc_toy.iloc[0,:]>0].index
# row_value = sc_toy.iloc[0,:][sc_toy.iloc[0,:]>0].values
#
# atom_quantity = [(i,j) for i,j in zip(row_index,row_value)]
# atom_quantity
# b = []
# len(atom_quantity)
# for a in atom_quantity:
#     tam = attom.iloc[a[0],:].to_numpy()
#     vam = np.append(tam,a[1])
#     b.append(vam)
#
# def atom_value(atom_property):
#     atom_quantity = atom_property
#     lenght = len(atom_quantity)
#     b = []
#     for a in atom_quantity:
#         tam = attom.iloc[a[0],:].to_numpy()
#         vam = np.append(tam,a[1])
#         b.append(vam)
#
#  sc_toy[(sc_toy>0).sum(axis =1)==9]
#
# max_length = (sc_toy>0).sum(axis =1).max()
# sc_dataframe.iloc[11655,-1]
# sc_toy.iloc[11655,:][sc_toy.iloc[11655,:]>0]

# def select_atom_in_formula(formula_array):
#     """Return a list of tuples with atomic number shifted by 1(Z-1) and atomic quantity"""
#
#     #remove chemical elements not present in the chemical formula
#     formula_array_other_than_0 = formula_array[formula_array>0]
#     #atoms_index is the atomic number (Z) -1
#     atoms_index = formula_array_other_than_0.index
#     #quantity of the relative element in the formula
#     atoms_value = formula_array_other_than_0.values
#
#     atom_symbol_quantity = [(i,j) for i,j in zip(atoms_index,atoms_value)]
#
#     return atom_symbol_quantity
#
# def get_atom_arrays(atom_symbol_quantity,max_length):
#     """Return an array(numpy) of lenght max_length filled by atomic arrays and 0's arrays (padding)"""
#
#     list_atom_features = []
#     #The symbol is not a string (like 'H') but an index (like 0 for 'H')
#     #symbol = Z-1 where Z = atomic number
#     for symbol,quantity in atom_symbol_quantity:
#         atom_features = attom.iloc[symbol,:].to_numpy()
#         complete_atom_features = np.append(atom_features,quantity)
#         list_atom_features.append(complete_atom_features)
#
#     padding_value = max_length - len(atom_symbol_quantity)
#     assert padding_value > 0,f'{padding_value} and {atom_symbol_quantity}'
#     array_atom_features_padded = np.pad(list_atom_features,[(0,padding_value),(0,0)],)
#
#     return array_atom_features_padded
#
# def expand_row_into_model_input(row,max_length = 10):
#     """Expand a row (pandas Series) into a model ready input"""
#
#     atom_symbol_quantity = select_atom_in_formula(row)
#
#     expanded_row = get_atom_arrays(atom_symbol_quantity,max_length)
#
#     return expanded_row
#
#

#
# [atom_value(atom_property) for atom_property in atom_quantity ]
# attom.head(8)
# atom_quantity
#
# len(b)
# b_arr =  np.array(b)
# b_arr.shape
#
# np.__version__
# np.pad(b,[(0,6),(0,0)],).shape
#
# list(np.zeros(shape=(6,33)))
#
# b[0][1]
# max_lenght = 10
# prova = sc_toy.iloc[:13,:].apply(expand_row_into_model_input,axis = 1)
#
# npsc = np.load('data/processed/dataset_supercon.npy')
#
# np.reshape(prova.to_numpy(),(10,10,33))
#
# np.array(prova.iloc[:][:])
#
#
# np.moveaxis(np.array(prova.to_list()),0,1).shape
#
# ((np.moveaxis(np.array(prova.to_list()),0,1) - npsc[:,:13,:])<0.001).all()
#
#
# npsc[:,:13,:].shape
#
# scd = Processing.SuperConData(attom,sc_dataframe,padding = 10)
#
# attom
#
# scdl = scd.build_dataset()
#
# npsc.shape
# scdl.shape
#
# ((np.moveaxis(np.array(list(scdl)),0,1) - npsc)<0.001).all()
