
#Classification
##Normalized data at 100
import pandas as pd
path = 'class_lin_data_test/'
metrics = pd.read_csv(path+'normalized/'+'class_score.csv')

#Risultati metriche per 10 classificatori
metrics.head(10)
#Valori medi delle metriche
metrics.mean()
#Deviazioni standard delle metriche
metrics.std()

#Predizioni sui quasicristalli
qc = pd.read_csv(path+'normalized/'+'quasi_crystall.csv',index_col='Name')

qc.head(23)
#Consistenza predizione sui quasi cristalli
qc_normalized = (qc>0.5).replace(True,1).sum(axis=1)
# Al63Cu24Fe13
qc_normalized
#Analisi predizione locale (sui singoli composti): 1 sc, 0 non testato, -1 non sc
test_local = pd.read_csv(path+'normalized/'+'test_dataframe.csv',index_col='material')

test_local

#Seleziona materiale che sono stati visti almeno una volta
tested_material = (test_local != 0).any(axis=1)

import numpy as np
#Conta il numero di volte in cui un materiale è stato testato
num_tested_material = test_local[tested_material].apply(np.abs).sum(axis=1)
#Somma le predizioni
prediction_on_material = test_local.sum(axis=1)

test_local['num_tested_material'] = num_tested_material
test_local['prediction_on_material'] = prediction_on_material
test_local.dropna(axis =0,inplace=True)
test_local
test_local.sort_values(by='num_tested_material',ascending=False)[['num_tested_material','prediction_on_material']].head(50)




#Classification
##Normale
import pandas as pd
path = 'class_lin_data_test/'
metrics = pd.read_csv(path+'total/'+'class_score.csv')

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
fig.suptitle('Linear Classification Metrics',fontsize=20)
fig.set_figwidth(10)
fig.set_figheight(10)

sns.boxplot(metrics['accuracy'],ax=ax[0,0])
sns.boxplot(metrics['precision'],ax=ax[0,1])
sns.boxplot(metrics['recall'],ax=ax[1,0])
sns.boxplot(metrics['f1'],ax=ax[1,1])
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

plt.savefig('Tesi_Linear_Classification_Metrics.png')
plt.show()
#%%
#Predizioni sui quasicristalli
qc = pd.read_csv(path+'total/'+'quasi_crystall.csv',index_col='Name')

qc.head(23)
#Consistenza predizione sui quasi cristalli
(qc>0.5).replace(True,1).sum(axis=1)# - qc_normalized

#Analisi predizione locale (sui singoli composti): 1 sc, 0 non testato, -1 non sc
test_local = pd.read_csv(path+'total/'+'test_dataframe.csv',index_col='material')

test_local

#Seleziona materiale che sono stati visti almeno una volta
tested_material = (test_local != 0).any(axis=1)

import numpy as np
#Conta il numero di volte in cui un materiale è stato testato
num_tested_material = test_local[tested_material].apply(np.abs).sum(axis=1)
#Somma le predizioni
prediction_on_material = test_local.sum(axis=1)
prediction_on_material
test_local['num_tested_material'] = num_tested_material
test_local['prediction_on_material'] = prediction_on_material
test_local.dropna(axis =0,inplace=True)
test_local
test_local.sort_values(by='num_tested_material',ascending=False)[['num_tested_material','prediction_on_material']].head(50)
test_local[['num_tested_material','prediction_on_material']].to_csv('test_differet_run.csv')
