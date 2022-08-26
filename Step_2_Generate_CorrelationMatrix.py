

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_corrgram.py

Scratch work on porting M. Friendly corrgram from R to python, applied to
interobserver similarity matrix from R. McLaughlin.

Takes a correlation or similarity type matrix and reorders the variables
according to the first two eigenvectors, to try to highlight blocks of
similar variables. Expects symmetric matrices.

Differs from most covariance structuring analysis by simply reordering
variables, as opposed to looking for latent factors through linear
combinations / rotations.

Clustering on the original variables and then displaying the similarities
in cluster order might produce comparable results, but this visualization
makes no prior assumptions on clustering / hierarchy.

See:
    
http://www.datavis.ca/papers/corrgram.pdf
https://github.com/kwstat/corrgram/blob/master/R/corrgram.r


The block we want to port is (line 262 to 274):

    # Calculate the size of the angle between the horizontal axis and the
    # PC vectors
    x.eigen <- eigen(cmat)$vectors[,1:2]
    e1 <- x.eigen[,1]
    e2 <- x.eigen[,2]
    alpha <- ifelse(e1>0, atan(e2/e1), atan(e2/e1)+pi) # Friendly eqn 1
    ord <- order(alpha)
    x <- if(type=="data") x[,ord] else x[ord, ord]
    cmat.return <- cmat.return[ord,ord]

Jason Ritt, Brown Univeristy
Initial: 20200926
"""

#%% Import block

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#%% Load data from terminal
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', type=str, dest='datafile',
                        action='store', required=True,
                        help='Absolute/relative path to the input data file')
    
    args = parser.parse_args()
    
    # load the data file
    df = pd.read_excel(args.datafile)
    similarity = df.iloc[:,1:].to_numpy()
#%% Load example data set from RM email, or simulated data


#if True:
    # True: Use real data
   # filename = '/Users/dowlettealameldin/Desktop/dhklab/StrokeProject/75_25_simularity_array.xlsx'
   # df = pd.read_excel(filename)
   # similarity = df.iloc[:,1:].to_numpy()
#else:
    # False: Make a random covariance (simularity) matrix
  #  signal = np.sin((6.28/5)*np.arange(16))
   # X = 2*(np.random.random((60,16))-0.5) + signal
   # X[20:35,:] = -X[20:35,:]
    #X = X[np.random.permutation(X.shape[0]),:]
   # similarity = np.cov(X)

plt.figure()
plt.imshow(similarity)
plt.axis('image')
plt.xlabel('Image number')
plt.ylabel('Image number')
plt.title('Similarities of image pairs: Original matrix')
plt.show()
plt.colorbar(similarity)
#%% Reorder variables by corrgram approach

# Idea is use the eigenvectors corresponding to the two largest eigenvalues
# to define an angular ordering, and then split that circle at the largest
# angle difference to sort the indices. See Friendly paper.
#
# Here we trust that unwrapping angles and sorting is good enough. Note
# one difference from the earlier implementation is the use of arctan2
# to get true, signed angles, rather than checking the sign of components
# and adding pi as needed.

_, eig_vecs = np.linalg.eigh(similarity)
# Get eigenvectors
# Using eigh as we are assuming we have a symmetric matrix.
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

inds_sort = angles.argsort()[-1::-1]
# (*) To change convention, keep original the sorting order.
angles_sort = angles[inds_sort]

angle_increments = np.diff(np.unwrap(np.append(angles_sort,angles_sort[0])))
# Need to wrap around index 0 in case that is the breakpoint, and unwrap
# to not pick up spurious jumps from crossing a multiple of pi.
start_ind = np.abs(angle_increments).argmax() + 1
# This is the breakpoint, as an index into **inds_sort**.

inds_ordered = np.append(inds_sort[start_ind:],inds_sort[0:start_ind])
# This is the variable ordering selected by the corrgram approach,
# as indices into the original variable set.

sorted_mat = similarity[:,inds_ordered]
sorted_mat = sorted_mat[inds_ordered,:]
#sorted_mat.to_csv('/Users/dowlettealameldin/Desktop/dhklab/StrokeProject/sortedmattest1.csv')
# Permute the columns and then the rows by the same indices to do a
# symmetric reordering.

plt.figure()
plt.imshow(sorted_mat)
plt.axis('image')
plt.xlabel('Image number')
plt.ylabel('Image number')
plt.title('Similarities of image pairs: Ordered matrix')
plt.show()
#fig.colorbar(sorted_mat, ax=ax1)
#plt.colorbar(sorted_mat)

print('The mapping from original image number to ordered position is')
print(np.array( [np.arange(len(inds_ordered)) , inds_ordered] ).T)
print('\n')
print('If LABELS is a list of strings for the experimental conditions')
print('in the **original** order, you can view the **sorted** conditions')
print('by LABELS[inds_ordered]')

# As loaded in this script:
#   df['Unnamed: 0'][inds_ordered]
