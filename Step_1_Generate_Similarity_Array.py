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

#upload the data you are going to turn into a similarity matrix
#the code is written to so that the first row of data is the third column in the excel sheet
#this code is also written so that it analyzes 7 columns (becuase there is 7 observers) column 2=observer 1...column8=observer 7
#make sure your data is formatted correclty before importing it so that the code reads to excel sheet correclty 
#data = pd.read_excel ('/Users/dowlettealameldin/Desktop/PhD/Stroke Project/RawDatasetWPython copy.xlsx')
#print(data)

# iterate through the dataframe to make a similarity matrix, i is the row
#
#this code functions by going through each row/column and completting pairwise comparisons

#this part is making the function that compares the image identities
def make_similarity_matrix(i, numObs, dataFile):
    '''
    Parameters
    ----------
    i : [int]
        row number (correlates to the images)
    numObs : [int]
        Number of observer columns in the data
    dataFile: [pandas dataframe]
	Data file to be used for calculating sums	

    Returns
    -------
    ColumnSum : completes pairwise comparisons
        (description)
    '''
    df_list = []
    for obs in range(numObs):
        #print('obs = ',obs+2)
        df = dataFile[dataFile.columns[obs+2]].eq(dataFile.iloc[i,obs+2])
        df *= 1
        df_T = df.T
        df_list.append(df_T)
    
    Column=pd.concat(df_list,axis=1)
    ColumnSum=Column.sum(axis=1)
    
    return ColumnSum



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', type=str, dest='datafile',
                        action='store', required=True,
                        help='Absolute/relative path to the input data file')
    
    args = parser.parse_args()
    
    # load the data file
    data = pd.read_excel(args.datafile)
    nCols = len(data.columns) - 2       # the first two columns are just names of the images
    
    #this part calls on the function made above to compile the results into a similarity matrix
    similaritymatrix = []
    for i in range(0, len(data)):
        similaritymatrix.append(make_similarity_matrix(i, nCols, data)) 
        print(similaritymatrix)
    
    #turn the array into a numpy array
    similarity_array = np.array(similaritymatrix)

    # make a dict for renaming the dataframe {col number : image name}
    nameList = list(data[data.columns[1]])
    length = range(len(nameList))
    names = dict(zip(length,nameList))

    #turn the array into a dataframe
    similarity_array = pd.DataFrame(similarity_array)

    similarity_array.rename(names, inplace=True, axis=0)
    similarity_array.rename(names, inplace=True, axis=1)

    # get the name of the input file to make a new output name
    fName = args.datafile.split('.')[0].split('/')[-1]
    
    #save the new similarity arry to an excel sheet
    similarity_array.to_excel('{}_Similarity_Matrix.xlsx'.format(fName))

    #plot
    plt.figure(figsize=(20,16)) #increased figure siz for poster
    sns.set(font_scale=5) #increased font for poster
    g = sns.heatmap(similarity_array, cmap='cividis', cbar=True)  #cmap dictates color palette

    plt.savefig('figures/{}_Similarity_Matrix.jpg'.format(fName))
    plt.savefig('figures/{}_Similarity_Matrix.pdf'.format(fName))
    plt.savefig('figures/{}_Similarity_Matrix.png'.format(fName))
    plt.savefig('figures/{}_Similarity_Matrix.tiff'.format(fName))
