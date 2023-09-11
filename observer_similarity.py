"""
Consolidating
TODO

Standard format is a Pandas dataframe whose first column is image names (can be
arbitrary) and remaining columns correspond to different observers. Rows are 
images, and the (i,n+1) element is the integer class label the nth observer gave
to the ith image.

It is allowable for the image names to contain duplicates, in which case they
are interpreted as categories (e.g. of experimental conditions). If image names
are unique, a Pandas series of length (number_of_images) can be used to compare
observer sorting to ground truth, if known.
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
        For input shape (nims,nobs), output will be (nims,nims+1). The (i,j+1)
        element counts how many of the nobs observers gave the ith and jth
        images the same label. There is no comparison across observers.
        If normalize is True, output have dtype float, otherwise int. The first
        column is always a copy of the image names from observer_labels
    
    Note: Code uses indexing rather than column and index names, because
    the latter do not have a reliable convention, and will be read from
    possibly non-standardized data files.
    
    Note: For simplicity the function just loops over the array and makes all
    pairwise comparisons. The similarity matrix must be symmetric, so half the
    comparisons are redundant and could be removed, but this is unlikely to
    have much impact at the input sizes typical of human observer labeling.
    '''
        
    nims = observer_labels.shape[0]
    similarity_matrix = np.full( (nims,nims), -1,dtype='int')
    for m in range(nims):
        similarity_matrix[m,:] = np.sum(
            observer_labels.iloc[m,1:] == observer_labels.iloc[:,1:], 
            axis=1 )
    if normalize:
        nobs = observer_labels.shape[1] - 1  # First col is image names
        similarity_matrix = similarity_matrix.astype('float') / nobs
    image_names = observer_labels.iloc[:,0]
    return image_names, similarity_matrix



def load_labels(filename, filetype=None, header_size=None):
    '''
    Loads observer labels from a CSV or Excel file. Assumes the data file is
    organized as a column of image names, followed by some number of columns
    of categorical integer labels provided by observers.

    Parameters
    ----------
    filename : str
        Path to CSV file containing observer labels
    filetype : None {default}, 'csv', 'xls', 'xlsx'
        If not None, forces attempt to load given file type, regardless of
        filename extension
    header_size : None, or int
        If None, will attempt to autodetect if a header is present. If int N, 
        will ignore N-1 rows of input file, and use the following row as
        column names.

    Returns
    -------
    pd.DataFrame
        First column is image names, remaining columns are integer observer
        labels.
        If the input had a header, it will be used to set column names
    '''
    try:
        pass
    except:
        raise 

    # Decide on load type
    # Load data (try block)
    # Detect header, set column names

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
    data.drop(columns="Image Number",inplace=True)
    #data.drop(columns="Unnamed: 9",inplace=True)
    data.drop(columns="Identitiy",inplace=True)

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
