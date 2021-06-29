#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 10:19:10 2021

@author: dowlettealameldin
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
#make sure there are no row/column headers in the data to use this code
dataframe=pd.read_excel('/Users/dowlettealameldin/Desktop/dhklab/StrokeProject/Manuscript/matrixalgebra/982_simularitymatrix_forpython.xlsx')



plt.figure(figsize=(20,16)) #increased figure siz for poster
sns.set(font_scale=5) #increased font for poster
g = sns.heatmap(dataframe, cmap='cividis') #cmap dictates color palette
plt.xlabel('Image number')
plt.ylabel('Image number')
plt.xticks(np.arange(10, 60, 10), ['10', '20', '30', '40','50','60'], rotation=0)
plt.yticks(np.arange(10, 60, 10), ['10', '20', '30', '40','50','60'], rotation=90)

#plt.savefig('DataSetXCorrelation_Matrix_MinusObsvFour.jpg')
plt.savefig('98_2_Simulated_Matrix_SimularityMatrix.tiff')


# In[3]:
    
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

#dataframe=pd.read_excel('/Users/dowlettealameldin/Desktop/dhklab/StrokeProject/DatasetY_Correlation_Matrix.xlsx')
dataframe=pd.read_excel('/Users/dowlettealameldin/Desktop/dhklab/StrokeProject/Manuscript/matrixalgebra/9010_simularitymatrix_forpython.xlsx')
print (dataframe)

plt.figure(figsize=(20,16)) #increased figure siz for poster
sns.set(font_scale=5) #increased font for poster
g = sns.heatmap(dataframe, cmap='cividis') #cmap dictates color palette

plt.xticks(np.arange(7, 60, 15), ['Ctrl', 'GD', 'OD', 'OGD'], rotation=0)
plt.yticks(np.arange(5, 60, 15), ['Ctrl', 'GD', 'OD', 'OGD'], rotation=90)

#plt.savefig('RandomSimulatedData_Correlation_Matrix.jpg')
plt.savefig('90_10_Simulated_Matrix_SimularityMatrix.tiff')
#plt.savefig('RandomSimulatedData_Correlation_Matrix.png')