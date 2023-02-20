# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data import make_dataset
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

ptable = make_dataset.get_periodictable()
rgr_series = pd.read_csv("../data/latent_space/rgr_avg_features.csv")["X"]
cls_series = pd.read_csv("../data/latent_space/cls_avg_features.csv")["X1"]
ptable["rgr_latent_space"] = rgr_series
ptable["cls_latent_space"] = cls_series

ptable.head()

ptable.dtypes

ptable.notnull().sum()

X = ptable.loc[:,(ptable.isnull().sum() < 1) & (ptable.dtypes != object)];
ycls = X["cls_latent_space"];
yrgr = X["rgr_latent_space"];
X = X.drop(["cls_latent_space","rgr_latent_space"],axis =1);
frgr = f_regression(X,yrgr);
fcls = f_regression(X,ycls);
#fs5 = SelectKBest(score_func = f_regression, k = 5)
#fs2 = SelectKBest(score_func = f_regression, k = 2)
#X_selected5_rgr = fs5.fit_transform(X, yrgr);
#X_selected2_rgr = fs2.fit_transform(X, yrgr);
#X_selected5_cls = fs5.fit_transform(X, ycls);
#X_selected2_cls = fs2.fit_transform(X, ycls);
X.columns

fcls

frgr

# +

for columns in X.columns:
    fig, axs = plt.subplots(ncols=2)
    sns.regplot(x=X[columns],y=ycls, ax=axs[0]);
    sns.regplot(x=X[columns],y=yrgr, ax=axs[1]);
# -

X = ptable[{"atomic_volume","melting_point","cls_latent_space","rgr_latent_space"}];
X = X.loc[X.atomic_volume.notnull()]
ycls = X["cls_latent_space"];
yrgr = X["rgr_latent_space"];
X = X.drop(["cls_latent_space","rgr_latent_space"],axis =1);
frgr = f_regression(X,yrgr);
fcls = f_regression(X,ycls);

fcls

frgr

for columns in X.columns:
    fig, axs = plt.subplots(ncols=2)
    sns.regplot(x=X[columns],y=ycls, ax=axs[0]);
    sns.regplot(x=X[columns],y=yrgr, ax=axs[1]);

X = ptable[{"density","melting_point","cls_latent_space","rgr_latent_space"}];
X = X.loc[X.density.notnull()]
ycls = X["cls_latent_space"];
yrgr = X["rgr_latent_space"];
X = X.drop(["cls_latent_space","rgr_latent_space"],axis =1);
frgr = f_regression(X,yrgr);
fcls = f_regression(X,ycls);

X

fcls

frgr

for columns in X.columns:
    fig, axs = plt.subplots(ncols=2)
    sns.regplot(x=X[columns],y=ycls, ax=axs[0]);
    sns.regplot(x=X[columns],y=yrgr, ax=axs[1]);

X = ptable[{"electron_affinity","melting_point","cls_latent_space","rgr_latent_space"}];
X = X.loc[X.electron_affinity.notnull()]
ycls = X["cls_latent_space"];
yrgr = X["rgr_latent_space"];
X = X.drop(["cls_latent_space","rgr_latent_space"],axis =1);
frgr = f_regression(X,yrgr);
fcls = f_regression(X,ycls);
X

fcls

frgr

for columns in X.columns:
    fig, axs = plt.subplots(ncols=2)
    sns.regplot(x=X[columns],y=ycls, ax=axs[0]);
    sns.regplot(x=X[columns],y=yrgr, ax=axs[1]);
axs[1].set_xlim(-2,+2)
axs[0].set_xlim(-2,+2)

X = ptable[{"evaporation_heat","melting_point","cls_latent_space","rgr_latent_space"}];
X = X.loc[X.evaporation_heat.notnull()]
ycls = X["cls_latent_space"];
yrgr = X["rgr_latent_space"];
X = X.drop(["cls_latent_space","rgr_latent_space"],axis =1);
frgr = f_regression(X,yrgr);
fcls = f_regression(X,ycls);
X

fcls

frgr

for columns in X.columns:
    fig, axs = plt.subplots(ncols=2)
    sns.regplot(x=X[columns],y=ycls, ax=axs[0]);
    sns.regplot(x=X[columns],y=yrgr, ax=axs[1]);

X = ptable[{"fusion_heat","melting_point","cls_latent_space","rgr_latent_space"}];
X = X.loc[X.fusion_heat.notnull()]
ycls = X["cls_latent_space"];
yrgr = X["rgr_latent_space"];
X = X.drop(["cls_latent_space","rgr_latent_space"],axis =1);
frgr = f_regression(X,yrgr);
fcls = f_regression(X,ycls);
X

fcls

frgr

for columns in X.columns:
    fig, axs = plt.subplots(ncols=2)
    sns.regplot(x=X[columns],y=ycls, ax=axs[0]);
    sns.regplot(x=X[columns],y=yrgr, ax=axs[1]);
axs[1].set_xlim(0,25)
axs[0].set_xlim(0,25)

X = ptable[{"group_id","melting_point","cls_latent_space","rgr_latent_space"}];
X = X.loc[X.group_id.notnull()]
ycls = X["cls_latent_space"];
yrgr = X["rgr_latent_space"];
X = X.drop(["cls_latent_space","rgr_latent_space"],axis =1);
frgr = f_regression(X,yrgr);
fcls = f_regression(X,ycls);
X

fcls

frgr

for columns in X.columns:
    fig, axs = plt.subplots(ncols=2)
    sns.regplot(x=X[columns],y=ycls, ax=axs[0]);
    sns.regplot(x=X[columns],y=yrgr, ax=axs[1]);

X = ptable[{"lattice_constant","melting_point","cls_latent_space","rgr_latent_space"}];
X = X.loc[X.lattice_constant.notnull()]
ycls = X["cls_latent_space"];
yrgr = X["rgr_latent_space"];
X = X.drop(["cls_latent_space","rgr_latent_space"],axis =1);
frgr = f_regression(X,yrgr);
fcls = f_regression(X,ycls);
X

fcls

frgr

for columns in X.columns:
    fig, axs = plt.subplots(ncols=2)
    sns.regplot(x=X[columns],y=ycls, ax=axs[0]);
    sns.regplot(x=X[columns],y=yrgr, ax=axs[1]);

X = ptable[{"specific_heat","melting_point","cls_latent_space","rgr_latent_space"}];
X = X.loc[X.specific_heat.notnull()]
ycls = X["cls_latent_space"];
yrgr = X["rgr_latent_space"];
X = X.drop(["cls_latent_space","rgr_latent_space"],axis =1);
frgr = f_regression(X,yrgr);
fcls = f_regression(X,ycls);
X

fcls

frgr

for columns in X.columns:
    fig, axs = plt.subplots(ncols=2)
    sns.regplot(x=X[columns],y=ycls, ax=axs[0]);
    sns.regplot(x=X[columns],y=yrgr, ax=axs[1]);
axs[1].set_xlim(0,1)
axs[0].set_xlim(0,1)

X = ptable[{"thermal_conductivity","melting_point","cls_latent_space","rgr_latent_space"}];
X = X.loc[(X.thermal_conductivity.notnull())&(X.thermal_conductivity<100)]
ycls = X["cls_latent_space"];
yrgr = X["rgr_latent_space"];
X = X.drop(["cls_latent_space","rgr_latent_space"],axis =1);
frgr = f_regression(X,yrgr);
fcls = f_regression(X,ycls);
X

fcls

frgr

X = ptable[{"thermal_conductivity","melting_point","cls_latent_space","rgr_latent_space"}];
X = X.loc[(X.thermal_conductivity.notnull())]
ycls = X["cls_latent_space"];
yrgr = X["rgr_latent_space"];
X = X.drop(["cls_latent_space","rgr_latent_space"],axis =1);
frgr = f_regression(X,yrgr);
fcls = f_regression(X,ycls);
X

fcls

frgr

for columns in X.columns:
    fig, axs = plt.subplots(ncols=2)
    sns.regplot(x=X[columns],y=ycls, ax=axs[0]);
    sns.regplot(x=X[columns],y=yrgr, ax=axs[1]);
axs[1].set_xlim(0,100)
axs[0].set_xlim(0,100)

X = ptable[{"en_pauling","melting_point","cls_latent_space","rgr_latent_space"}];
X = X.loc[X.en_pauling.notnull()]
ycls = X["cls_latent_space"];
yrgr = X["rgr_latent_space"];
X = X.drop(["cls_latent_space","rgr_latent_space"],axis =1);
frgr = f_regression(X,yrgr);
fcls = f_regression(X,ycls);
X

fcls

frgr

for columns in X.columns:
    fig, axs = plt.subplots(ncols=2)
    sns.regplot(x=X[columns],y=ycls, ax=axs[0]);
    sns.regplot(x=X[columns],y=yrgr, ax=axs[1]);

X = ptable[{"lattice_structure","cls_latent_space","rgr_latent_space"}];
X = X.loc[X.lattice_structure.notnull()]
catdata = X
yls = X["lattice_structure"];
X = X.drop(["lattice_structure"],axis =1);
f_rgrcls = f_classif(X,yls);
X

f_rgrcls

# +
catdata

sns.catplot(data=catdata, x="lattice_structure", y="rgr_latent_space")
sns.catplot(data=catdata, x="lattice_structure", y="cls_latent_space")
# -

X = ptable[{"block","cls_latent_space","rgr_latent_space"}];
catdata = X;
yls = X["block"];
X = X.drop(["block"],axis =1);
f_rgrcls = f_classif(X,yls);
X

f_rgrcls

# +
catdata

sns.catplot(data=catdata, x="block", y="rgr_latent_space")
sns.catplot(data=catdata, x="block", y="cls_latent_space")
# -


