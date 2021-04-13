import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Confront the element of the groups of the different rapresentation

DataA = pd.read_csv('src/study/mono_dim_data/mono_dim/mono_dim_rapp1.csv',index_col=0)
DataB = pd.read_csv('src/study/mono_dim_data/mono_dim/mono_dim_rapp2.csv',index_col=0)

#plot the rappresentation with the temperature to individuate the x range and the temperature
def plot_img(x,y):
    plt.plot(x,y,'ro')
    #plt.xticks(np.arange(0.6,1.4,0.2).tolist())
    plt.show()

plot_img(DataA.x,DataA.temp_pred)
plot_img(DataB.x,DataB.temp_pred)


#%%
#confront the different element in couple of group
A = find_group(DataA,-0.3,0.5,10)
B = find_group(DataB,-1,0,10)
len(A.intersection(B))/len(A)

len(A.intersection(B))/len(B)

len(A)
len(B)



#%%
#set of value that identify the group with high numbers of elements in common
[[A],[B]]
[[-4,2],[-5,-3]]
[[-0.3,0.5],[-1,0]]
[[2.5,3],[0,1]]
[[3.5,4.3],[0.7,1.1]]
[[2.7,3],[2,2.5]]
#%%
DataA
DataA.shape[0]
def find_group(Data,inf,sup,temp):
    group = set()
    for k in range(Data.shape[0]):
        if Data.x[k] > inf and Data.x[k] < sup:
            if Data.temp_oss[k] > temp:
                group.add(k)

    return group

#%%
import numpy as np
X = np.vstack([DataB.x,DataB.temp_oss])
X = np.moveaxis(X,0,1)
plot_img(X[:,0],X[:,1])
import seaborn as sns
sns.scatterplot(X[:,0],X[:,1])
from sklearn.cluster import KMeans
model = KMeans(n_clusters=2)
# fit the model
model.fit(X)
# assign a cluster to each example
y_t = model.predict(X)
from matplotlib import pyplot

sns.scatterplot(X[:,0],X[:,1],hue=y_t[:])


Y = []

for i in range(X.shape[0]):
    if y_t[i] ==1:
        Y.append([X[i,0],X[i,1]])

sns.scatterplot(Y[:,0],Y[:,1])


Y = np.array(Y)
Y.shape
sns.scatterplot(Y[:,0],Y[:,1],hue=yhat[:])

from sklearn.cluster import KMeans
model = KMeans(n_clusters=2)
# fit the model
model.fit(Y[:,0].reshape(-1, 1))
# assign a cluster to each example
yhat = model.predict(Y[:,0].reshape(-1, 1))
yhat.shape[0]
#%%

B_X =[]
for i in range(yhat.shape[0]):
    if yhat[i] == 1:
        B_X.append(Y[i,0])
B_X_max = np.array(B_X).max()
B_X_min = np.array(B_X).min()

#%%
B_index = []
X[:,0].shape[0]
for i in range(X[:,0].shape[0]):
    if X[i,0]>= B_X_min and X[i,0] <= B_X_max:
        B_index.append(i)

A_index = set(B_index)
# B_index = A1_index

len(A_index.intersection(B_index))/len(A_index)

len(A_index.intersection(B_index))/len(B_index)

B_index = A1_index
len(list(B_index))
len(list(A_index))
len(A_index.intersection(B_index))
AA = np.array(list(A_index)).sort()
BB = np.array(list(B_index)).sort()

AA == BB
#%%

sse = {}
for k in range(1, 5):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(Y)

    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np
#%%
fig, axs = plt.subplots(2)

fig.suptitle('Temperature predicted vs x')
axs[0].plot(DataA.x,DataA.temp_pred,'ro')
axs[1].plot(DataB.x,DataB.temp_pred,'ro')

#%%
mono_data_list = []
for i in range(10):
    mono_data_list.append(pd.read_csv('/home/claudio/aisc/project_aisc/src/study/mono_dim_data/mono_dim_10/'+'mono_dim_rapp_'+str(i)+'.csv',index_col=0))

#%%
#mono_data_list[0].x
#plot_img(mono_data_list[6].x,mono_data_list[6].temp_pred)
fig, axs = plt.subplots(2,5)

fig.suptitle('Predicted Temperature vs x')
for i in range(2):
    for j in range(5):
        axs[i,j].plot(mono_data_list[i+j].x,mono_data_list[i+j].temp_pred,'ro')
#%%
def kmean_display(i):
    X = np.vstack([mono_data_list[i].x,mono_data_list[i].temp_pred])
    X = np.moveaxis(X,0,1)

    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=2)

    model.fit(X)

    y_t = model.predict(X)
    from matplotlib import pyplot
    import seaborn as sns
    sns.scatterplot(X[:,0],X[:,1],hue=y_t[:])

    Y = []

    for i in range(X.shape[0]):
        if y_t[i] ==1:
            Y.append([X[i,0],X[i,1]])



    Y = np.array(Y)


    model = KMeans(n_clusters=2)

    model.fit(Y[:,0].reshape(-1, 1))

    yhat = model.predict(Y[:,0].reshape(-1, 1))
    #sns.scatterplot(Y[:,0],Y[:,1],hue=yhat[:])


kmean_display(0)
