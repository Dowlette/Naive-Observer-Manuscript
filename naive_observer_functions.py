"""
naive_observer_functions

Helper module for computing and analyzing naive observer labeling of images
into numbered categories.

Expects data to be in a CSV table whose first columns is image names, and 
remaining columns are integer labels given to the images by some number
of "naive" observers, who know only the number of categories but without
any particularly definition or ordering of categories.

The core function is make_similarity_matrix, which can be used without the
file load or plotting functions.
"""


#%% Import block

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import json, os, sys  # for parsing simulation config files


#%% Core functions for loading data, computing similarity, and plotting

def make_similarity_matrix(observer_labels, normalize=False):
    '''
    Parameters
    ----------
    observer_labels : pd.Dataframe
        rows are images, 1st col is image names, remaining cols are observers.
        Elements should be integer labels, with the (i,n+1) element stating the
        label the nth observer gave the ith image.
    normalize : Boolean (default False)
        If True, divide entries of similarity matrix by the number of observers,
        to return the fraction of common labels instead of absolute number.

    Returns
    -------
    pd.DataFrame
        For input shape (nims,nobs), output will be (nims,nims). The (i,j)
        element counts how many of the observers gave the ith and jth
        images the same label. The columns names are a copy of the image
        names from observer_labels, with the same datatype.
    
    If normalize is True, every entry is divided by number of rows (the
    number of observers), and the dtype is forced to float. Otherwise the 
    output is in counts, and the dtype is int.
        
    Note: Code uses indexing rather than column and index names, because
    the latter do not have a reliable convention, and will be read from
    possibly non-standardized data files.
    
    Note: For simplicity the function just loops over the array and makes all
    pairwise comparisons. The similarity matrix must be symmetric, so half the
    comparisons are redundant and could be removed, but this is unlikely to
    have much impact at the input sizes typical of human observer labeling.
    '''
        
    image_names = observer_labels.iloc[:,0]
    nims = len(image_names)

    similarity = np.full( (nims,nims), np.nan, dtype='float')
    for m in range(nims):
        similarity[m,:] = np.sum(
            observer_labels.iloc[m,1:] == observer_labels.iloc[:,1:], 
            axis=1 )

    if normalize:
        nobs = observer_labels.shape[1] - 1  # First col is image names
        similarity = similarity.astype('float') / nobs
        return pd.DataFrame(similarity, columns=image_names, dtype="float")
    else:
        return pd.DataFrame(similarity, columns=image_names, dtype="int")


def load_labels(filename, header=0, names=None):
    '''
    Loads observer labels from a CSV file. The data file should be organized
    as a column of image names, followed by some number of columns of 
    categorical integer labels provided by observers.

    Parameters
    ----------
    filename : str
        Path to CSV file containing observer labels
    header : int, list of int, None, default 'infer'
        Passed to pd.read_csv to control header/column name behavior.
        Default is to use file first row as column names.
    names: array-like, optional
        Passed to pd.read_csv to control header/column name behavior.
        To give custom column names, must pass names as a list. If overwriting
        names given in the file first row, set header=0. If the file first
        row is actually data, use header=None.

    Returns
    -------
    pd.DataFrame
        First column should be image names, remaining columns are integer 
        observer labels.
    '''
    # Note: originally planned to allow more heterogeneous input types,
    # but decided to force CSV, so is only a thin wrapper of read_csv.
    return pd.read_csv(filename, header=header, names=names)


def plot_similarity_matrix(df_similarity, image_names=None, figsize=None,
                           title=None, xlabel=None, ylabel=None,
                           cmap='cividis', cbar=False, **kwargs):
    '''
    Makes a heat map plot of similarity matrix.

    Parameters
    ----------
    df_similarity : pd.DataFrame
        Dataframe containing square matrix of similarity values
        Column names will be used as plot labels.
    image_names : optional list of strings
        If given, must have same length as number of images, and will be
        used to label ticks. If None, uses df_similarity column names.
    figsize: optional tuple
        Passed to plt.figure, see pyplot
    title, xlabel, ylabel: optional str
        If provided, will set the respective plot string
    Any remaining keyword parameters will be passed to heatmap function.

    Returns
    -------
    None.
    '''
    if image_names is None:
        image_names = df_similarity.columns
    plt.figure(figsize=figsize)
    sns.heatmap(df_similarity, cmap=cmap, cbar=cbar,
                xticklabels=image_names, yticklabels=image_names, **kwargs)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.axis('image')


#%% Corrgram functions

def corrgram_sort(df_similarity):
    '''
    Takes a correlation or similarity matrix and reorders the variables
    according to the first two eigenvectors, to try to highlight blocks of
    similar variables. Expects symmetric matrices.

    Parameters
    ----------
    df_similarity : pd.DataFrame
        Should be a square similarity matrix, with image names in the columns

    Returns
    -------
    pd.DataFrame
        Will have the same shape and values as the input, just rwith eordered
        columns and rows
    
    Differs from most covariance structuring analysis by simply reordering
    variables, as opposed to looking for latent factors through linear
    combinations / rotations. In the application to naive observer classifying,
    we want to consider permutations since the labels themselves have no
    intrinsic meaning, and no consistency across observers.
    
    Clustering on the original variables and then displaying the similarities
    in cluster order might produce comparable results, but this visualization
    makes no prior assumptions on clustering / hierarchy.
    
    Ported and modified from M. Friendly corrgram code in R.
    
    See:
        
    http://www.datavis.ca/papers/corrgram.pdf
    https://github.com/kwstat/corrgram/blob/master/R/corrgram.r
    '''
 
    # Idea is use the eigenvectors corresponding to the two largest eigenvalues
    # to define an angular ordering, and then split that circle at the largest
    # angle difference to sort the indices. See Friendly corrgram paper.
    #
    # Here we trust that unwrapping angles and sorting is good enough. Note
    # one difference from the earlier implementation is the use of arctan2
    # to get true, signed angles, rather than checking the sign of components
    # and adding pi as needed.

    # Using eigh as we are assuming we have a symmetric matrix.
    _, eig_vecs = np.linalg.eigh(df_similarity)

    # TODO: Should be the case that eigenvalues are sorted, so _last_ two
    # columns are largest. May need to check for actual maxima.
    angles = np.arctan2(eig_vecs[:,-2], eig_vecs[:,-1])

    # These are the angles (from the origin) in the plane if you plot the
    # components of the two eigenvectors against each other. In other words,
    # to each index in the similarity matrix we associate an angle.
    #
    # Now will order the variables by going clockwise(*) around the
    # plane, but need to pick a "breakpoint" to start, where there is a large
    # jump in angle.

    # (*) To change convention, delete the reverse indexing.
    inds_sort = angles.argsort()[-1::-1]
    angles_sort = angles[inds_sort]

    # Need to wrap around index 0 in case that is the breakpoint, and unwrap
    # to not pick up spurious jumps from crossing a multiple of pi.
    angle_increments = np.diff(np.unwrap(np.append(angles_sort,angles_sort[0])))
    
    # This is the breakpoint, as an index into **inds_sort**.
    start_ind = np.abs(angle_increments).argmax() + 1

    # This is the variable ordering selected by the corrgram approach,
    # as indices into the original variable set.
    inds_ordered = np.append(inds_sort[start_ind:],inds_sort[0:start_ind])

    # Permute the columns and then the rows by the same indices to do a
    # symmetric reordering. Note: simply sorting the values would lose
    # the label IDs used as column names.
    df_sorted = df_similarity.iloc[:,inds_ordered].copy()
    df_sorted = df_sorted.iloc[inds_ordered,:]

    return df_sorted


#%% Helper functions for config file parsing

# TODO: Do we need these? Environment requirement is way beyond 2.7 anyway

def AsciiEncodeDict(data):
    '''Encodes dict to ASCII for python 2.7'''
    # Will break in Python 3
    ascii_encode = lambda x: x.encode('ascii') if isinstance(x, unicode) else x
    return dict(map(ascii_encode, pair) for pair in data.items())

def OpenJSON(filename):
    '''Open JSON file as dictionary
    params:
	filename (str) = JSON file to open
    returns:
	dict: Python dictionary of JSON
    '''
    if sys.version_info.major == 3:
        return json.load(open(filename,'r'))
    else:
        return json.load(open(filename,'r'), object_hook=AsciiEncodeDict)


#%% Simulated observers

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
        
    def expose_config(self):
        # Just a debug helper function
        return self.config
    
    
    def createSimulatedMatrices(self):
        # loop through all desired tests
        for name in self.config.keys():
            path = self.config[name]['path']
            nRows = self.config[name]['nRows']
            nCols = self.config[name]['nCols']
            
            print('Creating file {}/{}_Simulated_Matrix...'.format(path,name))
            
            cur_sim_type = self.config[name]
            cur_p = cur_sim_type['p_control']
            label_ints = np.arange(len(cur_p)) + 1 # 1-offset by convention
            # TODO: Loop over treatment list, make general for number of classes
            #   Will require changing or dropping config file

            control_matrix = np.random.choice(label_ints, size=(nRows,nCols), p=self.config[name]['p_control'])

            gd_matrix = np.random.choice(label_ints, size=(nRows, nCols), p=self.config[name]['p_gd'])
            
            od_matrix = np.random.choice(label_ints, size=(nRows, nCols), p=self.config[name]['p_od'])
            
            ogd_matrix = np.random.choice(label_ints, size=(nRows, nCols), p=self.config[name]['p_ogd'])
            
            # create naive participant matrix
            naive_participant_matrix = np.concatenate((control_matrix, gd_matrix, od_matrix, ogd_matrix))
            
            # save into similarity matrix
            naive_participant_dataframe = pd.DataFrame(naive_participant_matrix)
            # add a blank column in the beginning to conform to standard data format
            naive_participant_dataframe.insert(loc=0, column='BLANK', value='')
            #naive_participant_dataframe.to_excel('{}/{}_Simulated_Matrix.xlsx'.format(path,name))
            # TODO: return matrix instead of write a file
            
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
        
        # TODO: Replace with general inputs
        #   How to read and use the right strings on tick marks?
        
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
        # TODO: replace with generic make_similarity_matrix
        # TODO: Change name to highlight looping over parameters
        
        for name in self.config.keys():
            path = self.config[name]['path']
            fileName = '{}/{}_Simulated_Matrix.xlsx'.format(path,name)
            # check if you've generated the simulated matrix excel sheet
            if not os.path.exists(fileName):
                print('{} does not exist, please run createMatrices() first.')
                continue
            else:
                print('Reading file {}/{}_Similarity_Matrix...'.format(path,name))
                # TODO: replace with arrays in memory
                # TODO: even with workaround, needs CSV not XLS
                df_raw = load_labels(fileName)
                #nCols = len(data.columns) - 2       # the first two columns are just names of the images
                #similarity_matrix = []
                #for i in range(len(data)):
                #    similarity_matrix.append(make_similarity_matrix(i, numObs=nCols, dataFile=data))
                #turn the array into a numpy array
                #similarity_array = np.array(similarity_matrix)
                #turn the array into a dataframe
                #similarity_df = pd.DataFrame(similarity_array)
                
                df_similarity = make_similarity_matrix(df_raw)
                
                #save the new similarity arry to an excel sheet
                # TODO: return result rather than write to file, handle multiple times
                # through loop
                
                #similarity_df.to_excel('{}/{}_Similarity_Matrix.xlsx'.format(path,name))
    
                # TODO: Plots should be specifically requested
                # now plot
                #self.plotSimilarityMatrices(arr   = df_similarity, 
                #                            title = name,
                #                            path  = 'figures/',
                #                            ext = 'png')


'''
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
'''


#%% Command line call

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', type=str, dest='datafile',
                        action='store', required=True,
                        help='Absolute/relative path to the input data file')
    args = parser.parse_args()
    
    # Load the data file request on the command line
    df_raw = load_labels(args.datafile)
    
    # Make similarity matrix
    df_similarity = make_similarity_matrix(df_raw)

    df_sorted = corrgram_sort(df_similarity)

    # Plot heatmap
    plot_similarity_matrix(df_similarity, title="Similarity of " + args.datafile)
    plot_similarity_matrix(df_sorted, title="Sorted Similarity of " + args.datafile)

    # -------- Using NO class
    # instantiate class with path to config file (default: NaiveObserver.json)
    npm = NaiveParticipantMatrix(args.config)
    
    # by default, we will assume the simulated matrices exist.
    if args.simulate:
        npm.createSimulatedMatrices()
    else:
        npm.createSimilarityMatrices()


    # Write output file with similarity matrix. Use input name as basename
    # for output file
    # TODO: Replace with os module calls
    # TODO: Write file in same directory as input? Use an input argument to
    #       override output location? Does not seem obviously correct to just
    #       write an output file to the current working directory. More
    #       generally I'm not a fan of code that writes new files without being
    #       super-transparent to user. I suggest default action might be just
    #       to create the heatmap, with no permanent outputs unless requested.
    #outname = args.datafile.split('.')[0].split('/')[-1]    
    #save_similarity_matrix(f'{outname}_Similarity_Matrix.csv',
    #                       similarity_matrix)

