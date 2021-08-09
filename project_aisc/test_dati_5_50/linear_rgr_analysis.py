#Regressione
##Normalized data at 100
import pandas as pd
path = 'rgr_lin_data_test/'
metrics = pd.read_csv(path+'normalized/'+'rgr_score.csv')

#Risultati metriche per 10 regressori
metrics.head(50)
#Valori medi delle metriche
metrics.mean()
#Deviazioni standard delle metriche
metrics.std()

#Predizioni sui quasicristalli
qc = pd.read_csv(path+'normalized/'+'quasi_crystall.csv',index_col='Name')

qc.head(23)
#Consistenza predizione sui quasi cristalli
qc_mean = qc.mean(axis=1)
qc_std = qc.std(axis=1)
qc_mean
qc_std
#Analisi predizione locale (sui singoli composti)
test_local = pd.read_csv(path+'normalized/'+'test_dataframe.csv',index_col='material')

test_local

#Seleziona materiale che sono stati visti almeno una volta
tested_material = (test_local != 0).any(axis=1)

import numpy as np
#Conta il numero di volte in cui un materiale è stato testato
test_local = test_local[tested_material]
num_tested_material = (test_local.apply(np.abs) > 0).sum(axis=1)

#Somma le predizioni
mean_prediction_on_material = test_local.replace(0,np.nan).mean(skipna=True,axis=1)
std_prediction_on_material = test_local.replace(0,np.nan).std(skipna=True,axis=1)
num_tested_material
test_local['num_tested_material'] = num_tested_material
test_local['mean_prediction_on_material'] = mean_prediction_on_material
test_local['std_prediction_on_material'] = std_prediction_on_material

test_local.dropna(axis =0,inplace=True)
test_local.sort_values(by='num_tested_material',ascending=False)
#[['num_tested_material','mean_prediction_on_material','std_prediction_on_material']].head(50)

#%%
import matplotlib.pyplot as plt
import seaborn as sns
#fig, ax = plt.plot()
plt.figure(figsize=(10,10))
sns.scatterplot(data = test_local,x='mean_prediction_on_material',y = 'std_prediction_on_material',sizes=(500,500))

#%%
predicted_values = test_local[test_local['num_tested_material'] > 10][['mean_prediction_on_material','std_prediction_on_material']]
#predicted_values = test_local[['mean_prediction_on_material','std_prediction_on_material']]
observed_values = pd.read_csv('/home/claudio/AISC/project_aisc/data/raw/supercon_tot.csv',index_col='material')
observed_values = observed_values['critical_temp']
observed_values.shape
predicted_values
observed_and_predicted_values = pd.concat([predicted_values,observed_values],join='inner',axis=1)
observed_and_predicted_values
import seaborn as sns
import matplotlib.pyplot as plt
#%%
#sns.scatterplot(data = observed_and_predicted_values,x='mean_prediction_on_material',y='critical_temp')
#plt.savefig('observed_and_predicted_values.png')
ax=plt.plot()
plt.errorbar(y='mean_prediction_on_material',x='critical_temp',yerr='std_prediction_on_material',data=observed_and_predicted_values,fmt='none')
plt.xlabel('Oberved critical temperature')
plt.ylabel('Predicted critical temperature')
#plt.savefig('observed_and_predicted_values_error_bar.png')


#%%
test_local.sort_values(by='std_prediction_on_material')[['num_tested_material','mean_prediction_on_material','std_prediction_on_material']].head(50)
import matplotlib.pyplot as plt
import matplotlib.image as image
#%%
fig, ax = plt.subplots(2,5,figsize=(50,15))
new_row = 0
for i in range(2):
    for j in range(5):
        im = image.imread(path+'total/rgr_img/predicted_vs_observed_regressor'+str(new_row)+'.png')
        ax[i,j].imshow(im)
        ax[i,j].set_frame_on(False)
        ax[i,j].xaxis.set_visible(False)
        ax[i,j].yaxis.set_visible(False)
        new_row +=1
#plt.savefig('linear_rgr_normalized_plots.png')
plt.show()


#%%
#Regressione
##Normale
import pandas as pd
path = 'rgr_lin_data_test/'
metrics = pd.read_csv(path+'total/'+'rgr_score.csv')

#Risultati metriche per 10 classificatori
metrics.head(10)
#Valori medi delle metriche
metrics.mean()
#Deviazioni standard delle metriche
metrics.std()
#%%
import seaborn as sns
import matplotlib.pyplot as plt
fig,ax = plt.subplots(2,2)
#ax.subplots_adjust(hspace=0.6, wspace=0.3)
fig.suptitle('Regression Metrics',fontsize=20)
fig.set_figwidth(10)
fig.set_figheight(10)

sns.boxplot(metrics['MSE'],ax=ax[0,0])
sns.boxplot(metrics['RMSE'],ax=ax[0,1])
sns.boxplot(metrics['MAE'],ax=ax[1,0])
sns.boxplot(metrics['R2'],ax=ax[1,1])
count = 0
for axes_i in ax:
    for ax_i in axes_i:
        for ind, label in enumerate(ax_i.get_xticklabels()):
            if count ==0 or count == 2:
                if ind % 2 == 0:
                    label.set_visible(True)
                else:
                    label.set_visible(False)
        count +=1

plt.savefig('Tesi_Linear_Regression_Metrics.png')
plt.show()

#%%
#Predizioni sui quasicristalli
qc = pd.read_csv(path+'total/'+'quasi_crystall.csv',index_col='Name')

qc.head(10)
#Consistenza predizione sui quasi cristalli
qc_mean = qc.mean(axis=1)
qc_std = qc.std(axis=1)
qc_mean
qc_std

#Analisi predizione locale (sui singoli composti): 1 sc, 0 non testato, -1 non sc
test_local = pd.read_csv(path+'total/'+'test_dataframe.csv',index_col='material')

test_local

#Seleziona materiale che sono stati visti almeno una volta
tested_material = (test_local != 0).any(axis=1)

import numpy as np
#Conta il numero di volte in cui un materiale è stato testato
test_local = test_local[tested_material]
num_tested_material = (test_local.apply(np.abs) > 0).sum(axis=1)

#Somma le predizioni
mean_prediction_on_material = test_local.replace(0,np.nan).mean(skipna=True,axis=1)
std_prediction_on_material = test_local.replace(0,np.nan).std(skipna=True,axis=1)
num_tested_material
test_local['num_tested_material'] = num_tested_material
test_local['mean_prediction_on_material'] = mean_prediction_on_material
test_local['std_prediction_on_material'] = std_prediction_on_material

test_local.dropna(axis =0,inplace=True)
test_local.sort_values(by='num_tested_material',ascending=False)[['num_tested_material','mean_prediction_on_material','std_prediction_on_material']].head(50)

test_local.sort_values(by='std_prediction_on_material')[['num_tested_material','mean_prediction_on_material','std_prediction_on_material']].head(50)
#%%

predicted_values = test_local[test_local['num_tested_material'] > 10][['mean_prediction_on_material','std_prediction_on_material']]
predicted_values = test_local[['mean_prediction_on_material','std_prediction_on_material']]
observed_values = pd.read_csv('/home/claudio/AISC/project_aisc/data/raw/supercon_tot.csv',index_col='material')
observed_values = observed_values['critical_temp']
observed_values.shape
predicted_values
observed_and_predicted_values = pd.concat([predicted_values,observed_values],join='inner',axis=1)
observed_and_predicted_values
import seaborn as sns
import matplotlib.pyplot as plt
#%%
#sns.scatterplot(data = observed_and_predicted_values,x='mean_prediction_on_material',y='critical_temp')
#plt.savefig('observed_and_predicted_values.png')
ax=plt.plot()
plt.errorbar(y='mean_prediction_on_material',x='critical_temp',yerr='std_prediction_on_material',data=observed_and_predicted_values,fmt='none')
plt.xlabel('Oberved critical temperature')
plt.ylabel('Predicted critical temperature')
plt.savefig('Tesi_linear_observed_and_predicted_values_error_bar.png')

#%%
import matplotlib.pyplot as plt
import matplotlib.image as image

#%%
plt.figure(figsize=(10,10))
ax = sns.scatterplot(data = test_local,x='mean_prediction_on_material',y = 'std_prediction_on_material',sizes=(500,500))
ax.set_xlabel('average prediction')
ax.set_ylabel('standard deviation prediction')
plt.savefig('Tesi_Linear_Regression_avg_std.png')
#%%
fig, ax = plt.subplots(2,5,figsize=(50,15))
new_row = 0
for i in range(2):
    for j in range(5):
        im = image.imread(path+'total/rgr_img/predicted_vs_observed_regressor'+str(new_row)+'.png')
        ax[i,j].imshow(im)
        ax[i,j].set_frame_on(False)
        ax[i,j].xaxis.set_visible(False)
        ax[i,j].yaxis.set_visible(False)
        new_row +=1
plt.savefig('linear_rgr_total_plots.png')
plt.show()
