#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 14:35:55 2022

@author: dowlettealameldin
"""

import numpy as np
import pandas as pd
import random


#25% chance they will be sorted into any group 
nums = np.random.choice([1,2,3,4], size=105, p=[.25, .25,.25,.25])
Control_Matrix=np.array(nums).reshape(15,7)

nums2 = np.random.choice([1,2,3,4], size=105, p=[.25, .25,.25,.25])
GD_Matrix=np.array(nums2).reshape(15,7)

nums3 = np.random.choice([1,2,3,4], size=105, p=[.25, .25,.25,.25])
OD_Matrix=np.array(nums3).reshape(15,7)

nums4 = np.random.choice([1,2,3,4], size=105, p=[.25, .25,.25,.25])
OGD_Matrix=np.array(nums4).reshape(15,7)

Naive_Particpant_Matrix=np.concatenate((Control_Matrix,GD_Matrix,OD_Matrix,OGD_Matrix))

#Save Matrix to make into simularity matrix
Naive_Particpant_Matrix = np.array(Naive_Particpant_Matrix)
Naive_Particpant_Matrix = pd.DataFrame(Naive_Particpant_Matrix)
Naive_Particpant_Matrix.to_excel('/Users/dowlettealameldin/Desktop/PhD/Stroke Project/25_25_25_25_Simulated_Matrix.xlsx')



# %%

#30% chance they will be sorted into any group 
nums = np.random.choice([1,2,3,4], size=105, p=[.3, .3,.2,.2])
Control_Matrix=np.array(nums).reshape(15,7)

nums2 = np.random.choice([1,2,3,4], size=105, p=[.3, .3,.2,.2])
GD_Matrix=np.array(nums2).reshape(15,7)

nums3 = np.random.choice([1,2,3,4], size=105, p=[.2, .2,.3,.3])
OD_Matrix=np.array(nums3).reshape(15,7)

nums4 = np.random.choice([1,2,3,4], size=105, p=[.2, .2,.3,.3])
OGD_Matrix=np.array(nums4).reshape(15,7)

Naive_Particpant_Matrix=np.concatenate((Control_Matrix,GD_Matrix,OD_Matrix,OGD_Matrix))

#Save Matrix to make into simularity matrix
Naive_Particpant_Matrix = np.array(Naive_Particpant_Matrix)
Naive_Particpant_Matrix = pd.DataFrame(Naive_Particpant_Matrix)
Naive_Particpant_Matrix.to_excel('/Users/dowlettealameldin/Desktop/PhD/Stroke Project/30_20_30_20_Simulated_Matrix.xlsx')



# %%

#40% chance they will be sorted into any group 
nums = np.random.choice([1,2,3,4], size=105, p=[.4, .4,.1,.1])
Control_Matrix=np.array(nums).reshape(15,7)

nums2 = np.random.choice([1,2,3,4], size=105, p=[.4, .4,.1,.1])
GD_Matrix=np.array(nums2).reshape(15,7)

nums3 = np.random.choice([1,2,3,4], size=105, p=[.1, .1,.4,.4])
OD_Matrix=np.array(nums3).reshape(15,7)

nums4 = np.random.choice([1,2,3,4], size=105, p=[.1, .1,.4,.4])
OGD_Matrix=np.array(nums4).reshape(15,7)

Naive_Particpant_Matrix=np.concatenate((Control_Matrix,GD_Matrix,OD_Matrix,OGD_Matrix))

#Save Matrix to make into simularity matrix
Naive_Particpant_Matrix = np.array(Naive_Particpant_Matrix)
Naive_Particpant_Matrix = pd.DataFrame(Naive_Particpant_Matrix)
Naive_Particpant_Matrix.to_excel('/Users/dowlettealameldin/Desktop/PhD/Stroke Project/40_10_40_10_Simulated_Matrix.xlsx')



# %%

#45% chance they will be sorted into any group 
nums = np.random.choice([1,2,3,4], size=105, p=[.45, .45,.05,.05])
Control_Matrix=np.array(nums).reshape(15,7)

nums2 = np.random.choice([1,2,3,4], size=105, p=[.45, .45,.05,.05])
GD_Matrix=np.array(nums2).reshape(15,7)

nums3 = np.random.choice([1,2,3,4], size=105, p=[.05, .05,.45,.45])
OD_Matrix=np.array(nums3).reshape(15,7)

nums4 = np.random.choice([1,2,3,4], size=105, p=[.05, .05,.45,.45])
OGD_Matrix=np.array(nums4).reshape(15,7)

Naive_Particpant_Matrix=np.concatenate((Control_Matrix,GD_Matrix,OD_Matrix,OGD_Matrix))

#Save Matrix to make into simularity matrix
Naive_Particpant_Matrix = np.array(Naive_Particpant_Matrix)
Naive_Particpant_Matrix = pd.DataFrame(Naive_Particpant_Matrix)
Naive_Particpant_Matrix.to_excel('/Users/dowlettealameldin/Desktop/PhD/Stroke Project/45_5_45_5_Simulated_Matrix.xlsx')



# %%

#48% chance they will be sorted into any group 
nums = np.random.choice([1,2,3,4], size=105, p=[.48, .48,.02,.02])
Control_Matrix=np.array(nums).reshape(15,7)

nums2 = np.random.choice([1,2,3,4], size=105, p=[.48, .48,.02,.02])
GD_Matrix=np.array(nums2).reshape(15,7)

nums3 = np.random.choice([1,2,3,4], size=105, p=[.02, .02,.48,.48])
OD_Matrix=np.array(nums3).reshape(15,7)

nums4 = np.random.choice([1,2,3,4], size=105, p=[.02, .02,.48,.48])
OGD_Matrix=np.array(nums4).reshape(15,7)

Naive_Particpant_Matrix=np.concatenate((Control_Matrix,GD_Matrix,OD_Matrix,OGD_Matrix))

#Save Matrix to make into simularity matrix
Naive_Particpant_Matrix = np.array(Naive_Particpant_Matrix)
Naive_Particpant_Matrix = pd.DataFrame(Naive_Particpant_Matrix)
Naive_Particpant_Matrix.to_excel('/Users/dowlettealameldin/Desktop/PhD/Stroke Project/28_2_48_2_Simulated_Matrix.xlsx')



# %%

#49% chance they will be sorted into any group 
nums = np.random.choice([1,2,3,4], size=105, p=[.49, .49,.01,.01])
Control_Matrix=np.array(nums).reshape(15,7)

nums2 = np.random.choice([1,2,3,4], size=105, p=[.49, .49,.01,.01])
GD_Matrix=np.array(nums2).reshape(15,7)

nums3 = np.random.choice([1,2,3,4], size=105, p=[.01, .01,.49,.49])
OD_Matrix=np.array(nums3).reshape(15,7)

nums4 = np.random.choice([1,2,3,4], size=105, p=[.01, .01,.49,.49])
OGD_Matrix=np.array(nums4).reshape(15,7)

Naive_Particpant_Matrix=np.concatenate((Control_Matrix,GD_Matrix,OD_Matrix,OGD_Matrix))

#Save Matrix to make into simularity matrix
Naive_Particpant_Matrix = np.array(Naive_Particpant_Matrix)
Naive_Particpant_Matrix = pd.DataFrame(Naive_Particpant_Matrix)
Naive_Particpant_Matrix.to_excel('/Users/dowlettealameldin/Desktop/PhD/Stroke Project/49_1_49_1_Simulated_Matrix.xlsx')

# %%

#49% chance they will be sorted into any group 
nums = np.random.choice([1,2,3,4], size=105, p=[.495, .495,.005,.005])
Control_Matrix=np.array(nums).reshape(15,7)

nums2 = np.random.choice([1,2,3,4], size=105, p=[.495, .495,.005,.005])
GD_Matrix=np.array(nums2).reshape(15,7)

nums3 = np.random.choice([1,2,3,4], size=105, p=[.005, .005,.495,.495])
OD_Matrix=np.array(nums3).reshape(15,7)

nums4 = np.random.choice([1,2,3,4], size=105, p=[.005, .005,.495,.495])
OGD_Matrix=np.array(nums4).reshape(15,7)

Naive_Particpant_Matrix=np.concatenate((Control_Matrix,GD_Matrix,OD_Matrix,OGD_Matrix))

#Save Matrix to make into simularity matrix
Naive_Particpant_Matrix = np.array(Naive_Particpant_Matrix)
Naive_Particpant_Matrix = pd.DataFrame(Naive_Particpant_Matrix)
Naive_Particpant_Matrix.to_excel('/Users/dowlettealameldin/Desktop/PhD/Stroke Project/49.5_.005_49.5_.005_Simulated_Matrix.xlsx')


# %%
nums = np.random.choice([1,2,3,4], size=105, p=[.7, .2,.05,.05])
Control_Matrix=np.array(nums).reshape(15,7)

nums2 = np.random.choice([1,2,3,4], size=105, p=[.2, .7,.05,.05])
GD_Matrix=np.array(nums2).reshape(15,7)

nums3 = np.random.choice([1,2,3,4], size=105, p=[.05, .05,.7,.2])
OD_Matrix=np.array(nums3).reshape(15,7)

nums4 = np.random.choice([1,2,3,4], size=105, p=[.05, .05,.2,.7])
OGD_Matrix=np.array(nums4).reshape(15,7)

Naive_Particpant_Matrix=np.concatenate((Control_Matrix,GD_Matrix,OD_Matrix,OGD_Matrix))

#Save Matrix to make into simularity matrix
Naive_Particpant_Matrix = np.array(Naive_Particpant_Matrix)
Naive_Particpant_Matrix = pd.DataFrame(Naive_Particpant_Matrix)
Naive_Particpant_Matrix.to_excel('/Users/dowlettealameldin/Desktop/PhD/Stroke Project/70_20_5_5_Simulated_Matrix.xlsx')


# %%
nums = np.random.choice([1,2,3,4], size=105, p=[.59, .39,.01,.01])
Control_Matrix=np.array(nums).reshape(15,7)

nums2 = np.random.choice([1,2,3,4], size=105, p=[.39, .59,.01,.01])
GD_Matrix=np.array(nums2).reshape(15,7)

nums3 = np.random.choice([1,2,3,4], size=105, p=[.01, .01,.59,.39])
OD_Matrix=np.array(nums3).reshape(15,7)

nums4 = np.random.choice([1,2,3,4], size=105, p=[.01, .01,.39,.59])
OGD_Matrix=np.array(nums4).reshape(15,7)

Naive_Particpant_Matrix=np.concatenate((Control_Matrix,GD_Matrix,OD_Matrix,OGD_Matrix))

#Save Matrix to make into simularity matrix
Naive_Particpant_Matrix = np.array(Naive_Particpant_Matrix)
Naive_Particpant_Matrix = pd.DataFrame(Naive_Particpant_Matrix)
Naive_Particpant_Matrix.to_excel('/Users/dowlettealameldin/Desktop/PhD/Stroke Project/59_39_1_1_Simulated_Matrix.xlsx')

# %%
nums = np.random.choice([1,2,3,4], size=105, p=[.85, .1,.025,.025])
Control_Matrix=np.array(nums).reshape(15,7)

nums2 = np.random.choice([1,2,3,4], size=105, p=[.1, .85,.025,.025])
GD_Matrix=np.array(nums2).reshape(15,7)

nums3 = np.random.choice([1,2,3,4], size=105, p=[.025, .025,.85,.1])
OD_Matrix=np.array(nums3).reshape(15,7)

nums4 = np.random.choice([1,2,3,4], size=105, p=[.025, .025,.1,.85])
OGD_Matrix=np.array(nums4).reshape(15,7)

Naive_Particpant_Matrix=np.concatenate((Control_Matrix,GD_Matrix,OD_Matrix,OGD_Matrix))

#Save Matrix to make into simularity matrix
Naive_Particpant_Matrix = np.array(Naive_Particpant_Matrix)
Naive_Particpant_Matrix = pd.DataFrame(Naive_Particpant_Matrix)
Naive_Particpant_Matrix.to_excel('/Users/dowlettealameldin/Desktop/PhD/Stroke Project/85_10_2.5_2.5_Simulated_Matrix.xlsx')

# %%
nums = np.random.choice([1,2,3,4], size=105, p=[.25, .25,.25,.25])
Control_Matrix=np.array(nums).reshape(15,7)

nums2 = np.random.choice([1,2,3,4], size=105, p=[.25, .25,.25,.25])
GD_Matrix=np.array(nums2).reshape(15,7)

nums3 = np.random.choice([1,2,3,4], size=105, p=[.25, .25,.25,.25])
OD_Matrix=np.array(nums3).reshape(15,7)

nums4 = np.random.choice([1,2,3,4], size=105, p=[.25, .25,.25,.25])
OGD_Matrix=np.array(nums4).reshape(15,7)

Naive_Particpant_Matrix=np.concatenate((Control_Matrix,GD_Matrix,OD_Matrix,OGD_Matrix))

#Save Matrix to make into simularity matrix
Naive_Particpant_Matrix = np.array(Naive_Particpant_Matrix)
Naive_Particpant_Matrix = pd.DataFrame(Naive_Particpant_Matrix)
Naive_Particpant_Matrix.to_excel('/Users/dowlettealameldin/Desktop/PhD/Stroke Project/25_25_25_25_Simulated_Matrix.xlsx')

# %%
nums = np.random.choice([1,2,3,4], size=105, p=[1, 0,0,0])
Control_Matrix=np.array(nums).reshape(15,7)

nums2 = np.random.choice([1,2,3,4], size=105, p=[0, 1,0,0])
GD_Matrix=np.array(nums2).reshape(15,7)

nums3 = np.random.choice([1,2,3,4], size=105, p=[0, 0,1,0])
OD_Matrix=np.array(nums3).reshape(15,7)

nums4 = np.random.choice([1,2,3,4], size=105, p=[0, 0,0,1])
OGD_Matrix=np.array(nums4).reshape(15,7)

Naive_Particpant_Matrix=np.concatenate((Control_Matrix,GD_Matrix,OD_Matrix,OGD_Matrix))

#Save Matrix to make into simularity matrix
Naive_Particpant_Matrix = np.array(Naive_Particpant_Matrix)
Naive_Particpant_Matrix = pd.DataFrame(Naive_Particpant_Matrix)
Naive_Particpant_Matrix.to_excel('/Users/dowlettealameldin/Desktop/PhD/Stroke Project/100_0_0_0_Simulated_Matrix.xlsx')


# %%

#import all the nessisary packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#upload the data you are going to turn into a simularity matrix
#the code is written to so that the first row of data is the third column in the excel sheet
#this code is also written so that it analyzes 7 columns (becuase there is 7 observers) column 2=observer 1...column8=observer 7
#make sure your data is formatted correclty before importing it so that the code reads to excel sheet correclty 
data = pd.read_excel ('/Users/dowlettealameldin/Desktop/PhD/Stroke Project/59_39_1_1_Simulated_Matrix.xlsx')
#print(data)

# iterate through the dataframe to make a simularity matrix, i is the row
#
#this code functions by going through each row/column and completting pairwise comparisons

#this part is making the function that compares the image identities
def make_simularity_matrix(i):
    df1= data[data.columns[2]].eq(data.iloc[i,2])
    df1=df1*1
    df1_T=df1.T
    df2 = data[data.columns[3]].eq(data.iloc[i,3])
    df2=df2*1
    df2_T=df2.T
    df3 = data[data.columns[4]].eq(data.iloc[i,4])
    df3=df3*1
    df3_T=df3.T
    df4 = data[data.columns[5]].eq(data.iloc[i,5])
    df4=df4*1
    df4_T=df4.T
    df5 = data[data.columns[6]].eq(data.iloc[i,6])
    df5=df5*1
    df5_T=df5.T
    df6= data[data.columns[7]].eq(data.iloc[i,7])
    df6=df6*1
    df6_T=df6.T
    df7= data[data.columns[8]].eq(data.iloc[i,8])
    df7=df7*1
    df7_T=df7.T
    #this code only goes through 7 columns (becuase there are 7 obersvers)
    #you can add to the code to allow it to iterate through more columns as nessisary
    
    Column=pd.concat([df1_T,df2_T,df3_T,df4_T,df5_T,df6_T,df7_T],axis=1)
   # Column=pd.concat([df1_T,df2_T,df3_T,df4_T,df5_T,df6_T],axis=1)
    ColumnSum=Column.sum(axis=1)
    return ColumnSum

#this part calls on the function made above to compile the results into a simularity matrix
simularitymatrix = []
for i in range(0, len(data)):
    simularitymatrix.append(make_simularity_matrix(i)) 
    print(simularitymatrix)
    
#turn the array into a numpy array
simularity_array = np.array(simularitymatrix)

#turn the array into a dataframe
simularity_array = pd.DataFrame(simularity_array)

#save the new simularity arry to an excel sheet
simularity_array.to_excel('/Users/dowlettealameldin/Desktop/PhD/Stroke Project/59_39_1_1_Simularity_Matrix.xlsx')

#plot



# %%

#import all the nessisary packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#upload the data you are going to turn into a simularity matrix
#the code is written to so that the first row of data is the third column in the excel sheet
#this code is also written so that it analyzes 7 columns (becuase there is 7 observers) column 2=observer 1...column8=observer 7
#make sure your data is formatted correclty before importing it so that the code reads to excel sheet correclty 
data = pd.read_excel ('/Users/dowlettealameldin/Desktop/PhD/Stroke Project/25_25_25_25_Simulated_Matrix.xlsx')
#print(data)

# iterate through the dataframe to make a simularity matrix, i is the row
#
#this code functions by going through each row/column and completting pairwise comparisons

#this part is making the function that compares the image identities
def make_simularity_matrix(i):
    df1= data[data.columns[2]].eq(data.iloc[i,2])
    df1=df1*1
    df1_T=df1.T
    df2 = data[data.columns[3]].eq(data.iloc[i,3])
    df2=df2*1
    df2_T=df2.T
    df3 = data[data.columns[4]].eq(data.iloc[i,4])
    df3=df3*1
    df3_T=df3.T
    df4 = data[data.columns[5]].eq(data.iloc[i,5])
    df4=df4*1
    df4_T=df4.T
    df5 = data[data.columns[6]].eq(data.iloc[i,6])
    df5=df5*1
    df5_T=df5.T
    df6= data[data.columns[7]].eq(data.iloc[i,7])
    df6=df6*1
    df6_T=df6.T
    df7= data[data.columns[8]].eq(data.iloc[i,8])
    df7=df7*1
    df7_T=df7.T
    #this code only goes through 7 columns (becuase there are 7 obersvers)
    #you can add to the code to allow it to iterate through more columns as nessisary
    
    Column=pd.concat([df1_T,df2_T,df3_T,df4_T,df5_T,df6_T,df7_T],axis=1)
   # Column=pd.concat([df1_T,df2_T,df3_T,df4_T,df5_T,df6_T],axis=1)
    ColumnSum=Column.sum(axis=1)
    return ColumnSum

#this part calls on the function made above to compile the results into a simularity matrix
simularitymatrix = []
for i in range(0, len(data)):
    simularitymatrix.append(make_simularity_matrix(i)) 
    print(simularitymatrix)
    
#turn the array into a numpy array
simularity_array = np.array(simularitymatrix)

#turn the array into a dataframe
simularity_array = pd.DataFrame(simularity_array)

#save the new simularity arry to an excel sheet
simularity_array.to_excel('/Users/dowlettealameldin/Desktop/PhD/Stroke Project/25_25_25_25_Simularity_Matrix.xlsx')

#plot



#%% make a figure
from matplotlib import pyplot as plt

#plt.imshow(simularity_array) # this is how you code for a simularity matrix to show up
#plt.xlabel("image #")
#plt.ylabel("image #")
#plt.title("83.3/16.6 Simulated Data")

plt.figure(figsize=(20,16)) #increased figure siz for poster
sns.set(font_scale=5) #increased font for poster
g = sns.heatmap(simularity_array, cmap='cividis') #cmap dictates color palette
plt.xticks(np.arange(7, 60, 15), ['Ctrl', 'GD', 'OD', 'OGD'], rotation=0)
plt.yticks(np.arange(7, 60, 15), ['Ctrl', 'GD', 'OD', 'OGD'], rotation=90)

plt.savefig('59_39_1_1_Simulated_Matrix_Simularity_Matrix.jpg')
#plt.savefig('90_10_Simulated_Matrix_Simularity_Matrix.pdf')
#plt.savefig('90_10_Simulated_Matrix_Simularity_Matrix.tiff')

