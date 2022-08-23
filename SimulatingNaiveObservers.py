#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 12:42:39 2022

@author: dowlettealameldin

Code for simulating naive observer sorting
"""
import os
import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt

# Helpers
from Step_1_Generate_Similarity_Array import make_similarity_matrix
from helpers import OpenJSON

# -------------------------------------------------------------------------
# Sorting functions
# -------------------------------------------------------------------------
class NaiveParticipantMatrix:
    '''
    Class to store, create, sort, and plot Naive Observer matrices
    '''
    def __init__(self, config):
        '''
        config : str
            Path to configuration file. Config file contains keys:
                MatrixName: name for output file
                |____ path:      (str)   path for writing output files (WIHTOUT trailing slash: '/')
                |____ nCols:     (int)   number of columns
                |____ nRows:     (int)   number of rows
                |____ p_control: (list)  list describing probabilities of sorting for control group
                |____ p_gd:      (list)  list describing probabilities of sorting for gd group
                |____ p_od:      (list)  list describing probabilities of sorting for od
                |____ p_ogd:     (list)  list describing probabilities of sorting for ogd
        '''
        self.config = OpenJSON(config)
        
    def createSimulatedMatrices(self):
        # loop through all desired tests
        for name in self.config.keys():
            path = self.config[name]['path']
            nRows = self.config[name]['nRows']
            nCols = self.config[name]['nCols']
            
            print('Creating file {}/{}_Simulated_Matrix...'.format(path,name))
            
            control = np.random.choice([1,2,3,4], size=nRows*nCols, p=self.config[name]['p_control'])
            control_matrix = np.array(control).reshape(nRows, nCols)

            gd = np.random.choice([1,2,3,4], size=nRows*nCols, p=self.config[name]['p_gd'])
            gd_matrix = np.array(gd).reshape(nRows, nCols)            
            
            od = np.random.choice([1,2,3,4], size=nRows*nCols, p=self.config[name]['p_od'])
            od_matrix = np.array(od).reshape(nRows, nCols)
            
            ogd = np.random.choice([1,2,3,4], size=nRows*nCols, p=self.config[name]['p_ogd'])
            ogd_matrix = np.array(ogd).reshape(nRows, nCols)    
            
            # create naive participant matrix
            naive_participant_matrix = np.concatenate((control_matrix, gd_matrix, od_matrix, ogd_matrix))
            
            # save into similarity matrix
            naive_participant_array = np.array(naive_participant_matrix)
            naive_participant_dataframe = pd.DataFrame(naive_participant_array)
            # add a blank column in the beginning to conform to standard data format
            naive_participant_dataframe.insert(loc=0, column='BLANK', value='')
            naive_participant_dataframe.to_excel('{}/{}_Simulated_Matrix.xlsx'.format(path,name))
            
    def plotSimilarityMatrices(self, arr, title, path, ext, figSize=(20,16),
                               fontSize=5, xlabel='image #', ylabel='image #',
                               cmap='cividis', xticks=['Ctrl', 'GD', 'OD', 'OGD'],
                               yticks=['Ctrl', 'GD', 'OD', 'OGD']):
        '''
            arr (np array):   input array to plot
            title (str):      title of plot
            path (str):       path to save it
            ext (str):        file extension (jpg, png, tiff...)
            figSize (tuple):  figure size 
            fontSize (int):   font size
            x/ylabel (str):   labels for x and y axes
            cmap (str):       colormap string
            x/yticks (list):  list of strings containing labels for the data
        '''
        plt.imshow(arr)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        
        plt.figure(figsize=figSize)
        sns.set(font_scale=fontSize)
        g = sns.heatmap(arr, cmap=cmap)
        # hard-code ranges for now..
        plt.xticks(np.arange(7,60,15), xticks, rotation=0)
        plt.yticks(np.arange(7,60,15), yticks, rotation=90)
        plt.savefig('{}/{}_Simulated_Similarity_Matrix.{}'.format(path,title,ext))
        
    
    def createSimilarityMatrices(self):
        for name in self.config.keys():
            path = self.config[name]['path']
            fileName = '{}/{}_Simulated_Matrix.xlsx'.format(path,name)
            # check if you've generated the simulated matrix excel sheet
            if not os.path.exists(fileName):
                print('{} does not exist, please run createMatrices() first.')
                continue
            else:
                print('Creating file {}/{}_Similarity_Matrix...'.format(path,name))
                data = pd.read_excel(fileName)
                nCols = len(data.columns) - 2       # the first two columns are just names of the images
                similarity_matrix = []
                for i in range(len(data)):
                    similarity_matrix.append(make_similarity_matrix(i, numObs=nCols, dataFile=data))
                #turn the array into a numpy array
                similarity_array = np.array(similarity_matrix)
                #turn the array into a dataframe
                similarity_df = pd.DataFrame(similarity_array)
                #save the new similarity arry to an excel sheet
                similarity_df.to_excel('{}/{}_Similarity_Matrix.xlsx'.format(path,name))
    
                # now plot
                self.plotSimilarityMatrices(arr   = similarity_array, 
                                            title = name,
                                            path  = 'figures/',
                                            ext = 'png')
                

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-c', type=str, dest='config', default='NaiveObserver_SimulationConfig.json',
                        action='store', required=False,
                        help='Absolute or relative path of JSON config file containing simulation info')
    parser.add_argument('--simulate', action='store_true',
                        help='Set this flag if you want/need to build the simulated matrices before plotting. Default False')
    
    args = parser.parse_args()
    
    # instantiate class with path to config file (default: NaiveObserver.json)
    npm = NaiveParticipantMatrix(args.config)
    
    # by default, we will assume the simulated matrices exist.
    if args.simulate:
        npm.createSimulatedMatrices()
    else:
        npm.createSimilarityMatrices()