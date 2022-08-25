# https://stackoverflow.com/a/68706205
# Workaround for handling missing system packages
options(repos = list(CRAN="http://cran.rstudio.com/"))
main <- function() {

  install.packages("readxl")

  library("readxl")
  #Load the xlsx file
  args <- commandArgs(trailingOnly = TRUE)
  filename <- args[1] #You should use simularity array or corrgram as datafile input (it will not change the PCA or K-Means output
  my_data <- read_excel(path = filename)
  #you should only have the names in the columns, no row names
  colnames(my_data) #identify the column names

  library(tidyverse)
  my_data2<-my_data %>% remove_rownames %>% column_to_rownames(var="names") #identify row names (no two row names can be the same)


  # PCA Variable Factor Map
  install.packages("FactoMineR",dependencies = TRUE, repos = "http://cran.us.r-project.org")
  library(FactoMineR)
  result <- PCA(my_data2) # graphs generated automatically

  ###% K-Means Clustering
  #Adapted from : https://afit-r.github.io/kmeans_clustering 

  #Install and Load packages 
  install.packages("tidyverse")
  install.packages("cluster")
  install.packages("factoextra")
  library(tidyverse)  # data manipulation
  library(cluster)    # clustering algorithms
  library(factoextra) # clustering algorithms & visualization
  install.packages("ggrepel")

  #Determine the optimal number of clusters to use with the Data using the WSS and Silhouette Method 
  set.seed(123)
  fviz_nbclust(my_data2, kmeans, method = "wss")
  fviz_nbclust(my_data2, kmeans, method = "silhouette")

  #Based on the output from above you change add or remove portions of the code below 
  options(ggrepel.max.overlaps = Inf)

  # this range can be changed depending on how many clusters you want
  for (i in 2:8) {
    # get the file name from the input argument
    path = filename
    if ( grepl(path, "/", fixed = TRUE) ) {    # user passed in a path to the file
      splitPath <- strsplit(path, split = "/")
      nameWithExt <- splitPath[[length(splitPath)]]
      splitName <- strsplit(nameWithExt, split = ".")
      name <- splitName[[1]]
    } else {   # user just passed the filename
      splitName <- strsplit(path, ".")
      name <- splitName[[1]]
    }
    fName <- sprintf("figures/%s_%i_K_Means_Clustering.png", name, i)
    # save to png 
    png(fName)
    kN <- kmeans(my_data2, centers = i, nstart = 25)
    pN <- fviz_cluster(kN, data = my_data2,repel = TRUE ,max.overlaps = Inf,labelsize = 8,main = "", palette = "jco", ggtheme = theme_bw())
    print(pN)
    dev.off()
  }

}

main()
