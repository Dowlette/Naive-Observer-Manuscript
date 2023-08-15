# Naive-Observer-Manuscript

The repo, a companion to manuscript [TODO](https://www.google.com), provides an open source pipeline to classify immunohistochemistry images by combining similarity judgments by independent human observers. 

## Usage

Ischemic brain injury occurs when there is impaired blood flow to the brain. Recently, our group has used a primary rat cortical spheroid model to study ischemic injury by depriving the spheroids of oxygen and/or glucose and examining the resulting morphological changes of capillary-like networks via confocal microscopy. However, qualitatively identifying morphological changes from spheroid images can be subjected to potential bias from the researcher, which could affect the interpretation of spheroid image data. To create a method for the classification of immunohistochemistry images, we present an alternative pipeline to standard image analysis and machine learning. Naïve Observers were blinded to the study and were instructed to classify unlabeled immunohistochemistry images into four groups based on the morphological features. Pairwise similarity comparisons of how each image was sorted were used to determine how similar morphological features in each image were to one another. Correlation matrices show that Naïve Observers were able to sort images into distinct groups. Further analysis showed that Naïve Observers were able to distinguish control and glucose deprived spheroids from oxygen deprived and oxygen-glucose deprived spheroids. This analysis method helps to identify potential phenotypes and provides the ability to analyze the perceived similarities between different images that may not be quantifiable using standard image analysis techniques. This analysis method has the potential for widespread use across labs with different in vitro models as it can be adapted to fit many different cell types of morphological data. Our code is written in Python, providing a platform that is free, open source, and easy to use. The Naïve Observer analysis method is versatile and can be applied to any imaging context. Moreover, our analysis pipeline is of particular interest to newly developing models where accurate feature selection may not be possible with traditional methods.

### Naive Observers

Naive Observers that were blinded to the experimental conditions separately attended four individual 15-minute video Zoom sessions. There was a 3-hour interval between sessions if they were held on the same day. Each of the four zoom sessions correlated to a different dataset. The Naïve Observers were given identical written instructions by the proctor before their first sorting session. During each session, Naïve Observers were instructed to download the folder with the images, sort them into four different groups, then reupload the folder to the Google Drive while under supervision of the proctor. Naïve Observers were instructed to sort the dataset into four groups based solely on the morphology and not based on “the size and shape of the spheroid, the intensity of the color of the image, or anything the Naïve Observers saw outside the perimeter of the spheroid that is present in the image”. Following the completion of all four sessions, the Naïve Observers were instructed to explain their sorting technique. Additionally, the data was formatted as in the folders data > raw datasets. 

### Step 1 - Generate Similarity Matrix

For each dataset, each Naïve Observer assigned every image to a folder (Groups 1-4). Each image was tagged with a unique sorting identity compiled from all 7 Observers. Since group designations were arbitrarily assigned by the Naïve Observers, we analyzed images using their compiled sorting identity from all 7 Naïve Observers. This code was written in Python and was used to compare the different images. This code functions by scoring the frequency in which two images were sorted into the same Groups across the Observers. This process was repeated using the code to obtain a symmetrical similarity matrix. 

### Step 2 - Generate Corrgram

In order to identify which images were most similar to one another, the similarity matrix was reordered according to the first two eigenvectors. This code is based on the Friendly et al., 2002 paper. 

### Step 3 - (Optional)

This code calculates the PCA and K-Means Clustering of the similiary matrix or correlation matrix in order to identify which images were simular to one another using the file `Step_3_optional_PCA_K_MEANS.R`. Please note that the input data must be formatted so that there is one column to the left most of the data titled 'names' identifying the images. Run this step by typing: `Rscript Step_3_optional_PCA_K_MEANS.R </path/to/formated/data/file>` to generate the K-Means and PCA graphs, storing the figures in the `figures/` directory by default. 

### Step 4 - (Optional)

In order to determine the discriminability between images you can simulate data. This code simulates data using the file `SimulatingNaiveObservers.py`. By default, the script reads from the JSON config file `NaiveObserver_SimulationConfig.json`. You can pass a different configuration file with the flag `-c </path/to/config/file>`. The config file should have the format:
```json
"name": (str) name for the simulated dataset + simularity matrix
{ 
  "path": (str) path for saving simulated dataset + similarity matrix (without trailing backslash '/'),
  "nCols": (int) number of columns (representing total number of participants) to create for the simulated dataset,
  "nRows": (int) number of rows (representing total number of images in each group) to create for the simulated dataset,
  "p_control": (list) the probability distrobutions for sorting the controls,
  "p_gd": (list),
  "p_od": (list),
  "p_ogd": (list)
},
```
Note that, for the `p_*` keys, the first item represents the probability an observer is sorted into the control group, the second the glucose deprivation group, the third the oxygen deprivation group, and the fourth the oxygen-glucose deprivation group. 

Run this step first by `python SimulatingNaiveObservers.py --simulate` to generate the simulated datasets described in the config file. Then run `python SimulatingNaiveObservers.py` by itself to generate the similarity matrices and plot the results, storing the figures in the `figures/` directory by default.


## Installation

For Mac/Linux Systems: 

First set up the python virtual environment. From a terminal (Mac, Linux): 

```shell
conda env create -f environment.yml
```

In the terminal you can then activate the environment with 

```shell
conda activate spheroid_images
```

For windows

Go to windows power shell
