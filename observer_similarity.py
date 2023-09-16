"""
naive_observer_analysis

Module for computing and analyzing naive observer labeling of images into
numbered categories.

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


#%%

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


def save_similarity_matrix(filename, similarity_matrix, filetype=None, 
                           header_size=None):
    '''
    

    Parameters
    ----------
    filename : str
        Path to CSV file containing observer labels
    similarity_matrix : square array like
        TODO: How to handle categorical vs unique image names
    filetype : None {default}, 'csv', 'xls', 'xlsx'
        If not None, forces attempt to load given file type, regardless of
        filename extension
    header_size : None, or int
        If None, will attempt to autodetect

    Returns
    -------
    None.
    '''
    pass

    # Decide on load type
    # Load data (try block)
    # Detect header, set column names


def show_similarity_matrix(image_names, similarity_matrix, figsize=None,
                           cmap='cividis', cbar=False):
    '''
    Makes a heat map plot of similarity matrix.

    Parameters
    ----------
    similarity_matrix : square array like
        Square matrix of similarity values
        If a pd.DataFrame, the column  names will be used as plot labels.
        If a numpy array, the plot labels will be integers.
    TODO :  Other input parameters
    
    Returns
    -------
    None.
    '''
    plt.figure(figsize=figsize)
    sns.heatmap(similarity_matrix, cmap=cmap, cbar=cbar,
                xticklabels=image_names, yticklabels=image_names)
    plt.axis('image')

def simulate_similarity_matrix():
    pass



#%% Temporary data load for debuging

if 0:
    #cd /Users/jritt/Code/McLaughlin/Naive-Observer-Manuscript
    #datafile = 'test_raw_data.csv'
    datafile = 'datasets/raw/RawDatasetZ.csv'
    data = pd.read_csv(datafile)
    data.drop(columns="Image ID",inplace=True)
    #data.drop(columns="Unnamed: 9",inplace=True)
    #data.drop(columns="Identitiy",inplace=True)

# TODO: Check image number consistency across datasets
# TODO: Include code file for true image type by identifier

#%%


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', type=str, dest='datafile',
                        action='store', required=True,
                        help='Absolute/relative path to the input data file')
    args = parser.parse_args()
    
    # Load the data file request on the command line
    data = load_labels(args.datafile)
    
    # Make similarity matrix
    similarity_matrix = make_similarity_matrix(data)

    # Write output file with similarity matrix. Use input name as basename
    # for output file
    # TODO: Replace with os module calls
    # TODO: Write file in same directory as input? Use an input argument to
    #       override output location? Does not seem obviously correct to just
    #       write an output file to the current working directory. More
    #       generally I'm not a fan of code that writes new files without being
    #       super-transparent to user. I suggest default action might be just
    #       to create the heatmap, with no permanent outputs unless requested.
    outname = args.datafile.split('.')[0].split('/')[-1]    
    save_similarity_matrix(f'{outname}_Similarity_Matrix.csv',
                           similarity_matrix)

    # Plot
    show_similarity_matrix(similarity_matrix, figsize=(20,16))
