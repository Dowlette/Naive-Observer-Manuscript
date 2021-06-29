#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 14:36:58 2020

@author: dowlettealameldin
"""



#import all the nessisary packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#upload the data you are going to turn into a simularity matrix
#the code is written to so that the first row of data is the third column in the excel sheet
#this code is also written so that it analyzes 7 columns (becuase there is 7 observers) column 2=observer 1...column8=observer 7
#make sure your data is formatted correclty before importing it so that the code reads to excel sheet correclty 
data = pd.read_excel ('/Users/dowlettealameldin/Desktop/dhklab/StrokeProject/98_2_Simulated_Matrix.xlsx')
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
simularity_array.to_excel('/Users/dowlettealameldin/Desktop/dhklab/StrokeProject/SimulatedData/98_2_Simularity_Matrix.xlsx')

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

plt.savefig('90_10_Simulated_Matrix_Simularity_Matrix.jpg')
plt.savefig('90_10_Simulated_Matrix_Simularity_Matrix.pdf')
plt.savefig('90_10_Simulated_Matrix_Simularity_Matrix.tiff')



