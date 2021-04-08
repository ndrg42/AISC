import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Confront the element of the groups of the different rapresentation

DataA = pd.read_csv('mono_dim_data/mono_dim/mono_dim_rapp1.csv',index_col=0)
DataB = pd.read_csv('mono_dim_data/mono_dim/mono_dim_rapp2.csv',index_col=0)

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
