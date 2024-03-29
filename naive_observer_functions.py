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

import json, sys  # for parsing simulation config files


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

def _AsciiEncodeDict(data):
    '''Encodes dict to ASCII for python 2.7'''
    # Will break in Python 3
    ascii_encode = lambda x: x.encode('ascii') if isinstance(x, unicode) else x
    return dict(map(ascii_encode, pair) for pair in data.items())

def _OpenJSON(filename):
    '''Open JSON file as dictionary
    params:
	filename (str) = JSON file to open
    returns:
	dict: Python dictionary of JSON
    '''
    if sys.version_info.major == 3:
        return json.load(open(filename,'r'))
    else:
        return json.load(open(filename,'r'), object_hook=_AsciiEncodeDict)

def get_config(filepath):
    return _OpenJSON(filepath)


#%% Simulate observers with specified classification ability

def create_simulated_matrices(config):
    '''
    Returns a dictionary of simulated similarity matrices (or just a numpy
    array if config has only one experiment in it). 
 
    Parameters
    ----------
    config : dictionary with the following structure
        path:           (str)   path for writing output files
        num_observers:  (int)   number of observers to use when generating samples
        num_images_per_class:     (int)   number of images _within_ each class
        experiments: dict
        
        The experiments dict should have strings for keys, which are taken as
        the names of the experiments. Each value should be another dict, with
        strings as keys that are taken as treatment names. The corresponding
        values should be an array of floats that add to one, with the same
        length as the number of treatments/keys.
        The floats are the probabilities that an image of the given treatment
        type will be placed into the different class numbers.

    It is intended but not required that config be loaded from a JSON 
    configuration file:
        config = nof.get_config(filepath)
        simulated_similarity = nof.create_simulated_matrices(config)

    Returns
    -------
    dict of pd.DataFrames
        The keys will be experiment names, and the DataFrames will use
        treatments as the column names of similarity matrices (the outputs
        of make_similarity_matrix)
    '''

    nRows = config['num_images_per_class']
    nCols = config['num_observers']

    simulated_similarity = dict()

    # loop through all desired tests
    experiments = config['experiments']
    for exp_name in experiments:
        
        print('Creating {} simulated matrix...'.format(exp_name))
        
        cur_exp = experiments[exp_name]
        num_classes = len( cur_exp.keys() )
        class_ints = np.arange(num_classes) + 1 # 1-offset by convention

        subject_labels = np.full( (nRows*num_classes, nCols), np.nan)
        df_labels = pd.DataFrame(subject_labels)
        treat_ids = []
        for treat_ind, treat_name in enumerate(cur_exp):
            block_start = treat_ind*nRows
            block_end = (treat_ind+1)*nRows
            df_labels.iloc[block_start:block_end,:] = np.random.choice( 
                                       class_ints, size=(nRows,nCols),
                                       p=cur_exp[treat_name] ).astype(int)
            treat_ids = treat_ids + [treat_name]*nRows
        
        # Add the names of treatments to first column
        df_labels.insert(loc=0, column='Treatments', value=treat_ids)
        
        # TODO: Break into two functions: simulate_labels, then similarity?

        # Compute similarity matrix
        simulated_similarity[exp_name] = make_similarity_matrix( df_labels )

    return simulated_similarity



#%% Command line call


def _main(args):
    '''
    Intended to be called only from if __main__ 
    args is usually an argparse.Namespace obejct, but can be anything
    with the correct attributes if passed in manually
    '''
    
    if args.datafile:
        # Load the data file request on the command line
        df_raw = load_labels(args.datafile)
        
        # Make similarity matrix and sorted version
        df_similarity = make_similarity_matrix(df_raw)
        df_sorted = corrgram_sort(df_similarity)
    
        # Plot heatmaps
        plot_similarity_matrix(df_similarity, title="Similarity of " + args.datafile)
        plot_similarity_matrix(df_sorted, title="Sorted Similarity of " + args.datafile)
        
    if args.configfile:
        # Simulate observers
        config = get_config(args.configfile)
        dict_simulated = create_simulated_matrices(config)
    
        for exp_name in dict_simulated:
            df_cur = dict_simulated[exp_name]
            df_cur_sorted = corrgram_sort(df_cur)
            plot_similarity_matrix(df_cur, title="Similarity of " + exp_name)
            plot_similarity_matrix(df_cur_sorted, title="Sorted Similarity of " + exp_name)

    print(" ")
    print("Displaying all requested figures.")
    print("Closing all plot windows will quit.")
    print("Hitting Ctrl-C in the terminal should close all and quit, but due")
    print("to Matplotlib's buggy cross-platform performance, may fail.")
    print("Recommended usage is to import this moodule in your own code, rather")
    print("than call from a CLI.")    
    plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(epilog='Functions for naive observer analysis. ' + 
                            'See make_similarity_matrix and create_simulated_matrices ' +
                            'for expected input and configuration formats.')
    parser.add_argument('-d', type=str, dest='datafile',
                        action='store', required=False,
                        help='Absolute/relative path to a data file')
    parser.add_argument('-c', type=str, dest='configfile',
                        action='store', required=False,
                        help='Absolute/relative path to a config file')
    args = parser.parse_args()

    if ( (args.datafile==None) and (args.configfile==None) ):
        print('At least one of a data file or a config file path should be given')
        print('when calling from a command line (recommended usage is to import')
        print('in your own code with handling of filenames and save options)')
        print(' ')
        parser.print_help()
        # SystemExit is annoying in REPL, but we should land here only if
        # actually invoked as a script
        raise SystemExit()
    
    _main(args)
