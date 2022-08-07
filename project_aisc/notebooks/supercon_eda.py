# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SuperCon EDA

# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# %%
#Load SueprCon data
supercon_path = "../data/raw/supercon.csv"
supercon = pd.read_csv(supercon_path)

# %%
supercon

# %% [markdown]
# # Critical Temperature
#
# we inspect critical temperature distribution, its shape, center, spread and position.

# %%
critical_temperature = supercon['critical_temp']
critical_temperature.hist()

# %%
sns.boxplot(x = critical_temperature,whis=1.5)

# %% [markdown]
# It's right skewed distribution, most of the samples are under Tc:60 K; we don't expect regressor to be reliable for materials above ~100 K. We have to point out that critical temperature is bounded below by 0 K; if a given material IS a supercondutor and the ML model predict a critical temperature above 100 K, we know that is is at least 100 K but we are not sure about the real value. 

# %%
print(f"median : {critical_temperature.median()}\nmin & max temperature :{critical_temperature.min(),critical_temperature.max()}\n25 & 75 percentile: {np.percentile(critical_temperature,[25,75])}")

# %% [markdown]
# # Elements

# %%
supercon_elements = supercon.iloc[:,:-2]
# We inspect the distribution of elements on superconducting materials
supercon_elements.sum().sort_values(ascending=False).plot.bar(figsize=(20,20))

# %%
supercon_elements.sum().sort_values(ascending=False).head(16)

# %%
# NOw we look at the percentage of elements that appear on a chemical formula at least one
((supercon_elements>0).sum()/supercon_elements.shape[0]).sort_values(ascending=False).plot.bar(figsize= (20,20))


# %%
((supercon_elements>0).sum()/supercon_elements.shape[0]).sort_values(ascending=False).head(15)


# %%
# We compute the average number of atoms for chemical formula
(supercon_elements.sum()/(supercon_elements>0).sum()).sort_values(ascending=False).head(15)


# %%
most_common_elements = list(((supercon_elements>0).sum()/supercon_elements.shape[0]).sort_values(ascending=False).head(10).index)
supercon_elements[most_common_elements].corr().style.background_gradient()

# %%
(supercon_elements[most_common_elements]>0).corr().style.background_gradient()

# %% [markdown]
# Oxygen is the most present element both in percentage of compounds and in absolute number of atoms, showing an average of 6 oxygen atoms per material. Nb is very present in the dataset but only in absolute number and not in percentage of materials, in fact it has an average quantity greater than oxygen but in the percentage of materials it is not even among the top 10. Other elements very present both in percentage of materials and in absolute number they are Cu, Ba, Sr, Ca. The elements Cu, Ba, Sr, Ca are strongly correlated with the presence of oxygen, to a lesser extent Y and Ba, not at all Fe, La and As. Cu is correlated to the presence of Ba, Sr, ca and Y; Ba with Y; Sr and Ca with Bi. There is a strong correlation between the presence of Fe and As.

# %%
