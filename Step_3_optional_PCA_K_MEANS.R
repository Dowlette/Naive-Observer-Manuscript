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

  kList <- list()
  pList <- list()
  for (i in 2:8) {
    kN = kmeans(my_data2, centers = i, nstart = 25)
    pN = fviz_cluster(k2, data = my_data2,repel = TRUE ,max.overlaps = Inf,labelsize = 8,main = "", palette = "jco", ggtheme = theme_bw())
    kList.append(kN)
    pList.append(pN)
  }
  
  #Plotting the outputs
  library(gridExtra)
  
  for (p in pList) {
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
     fName = sprintf("%s_K_Means_Clustering.png", name)
     # save the file 
     ggsave(file=fName, p)
  }
  
  #If your optimal number of clusters is 2, run the section below
  k2 <- kmeans(my_data2, centers = 2, nstart = 25)
  p2 <- fviz_cluster(k2, data = my_data2,repel = TRUE ,max.overlaps = Inf,labelsize = 8,main = "", palette = "jco", ggtheme = theme_bw())

  #If your optimal number of clusters is 3, run the section below
  k3 <- kmeans(my_data2, centers = 3, nstart = 25)
  p3 <- fviz_cluster(k3, data = my_data2,repel = TRUE ,max.overlaps = Inf,labelsize = 8,main = "", palette = "jco", ggtheme = theme_bw())

  #If your optimal number of clusters is 4, run the section below
  k4 <- kmeans(my_data2, centers = 4, nstart = 25)
  p4 <- fviz_cluster(k4, data = my_data2,repel = TRUE ,max.overlaps = Inf,labelsize = 8,main = "", palette = "jco", ggtheme = theme_bw())


  #If your optimal number of clusters is 5, run the section below
  k5 <- kmeans(my_data2, centers = 5, nstart = 25)
  p5 <- fviz_cluster(k5, data = my_data2,repel = TRUE ,max.overlaps = Inf,labelsize = 8,main = "", palette = "jco", ggtheme = theme_bw())

  #If your optimal number of clusters is 6, run the section below
  k6 <- kmeans(my_data2, centers = 6, nstart = 25)
  p6 <- fviz_cluster(k6, data = my_data2,repel = TRUE ,max.overlaps = Inf,labelsize = 8,main = "", palette = "jco", ggtheme = theme_bw())

  #If your optimal number of clusters is 7, run the section below
  k7 <- kmeans(my_data2, centers = 7, nstart = 25)
  p7 <- fviz_cluster(k7, data = my_data2,repel = TRUE ,max.overlaps = Inf,labelsize = 8,main = "", palette = "jco", ggtheme = theme_bw())

  #If your optimal number of clusters is 8, run the section below
  k8 <- kmeans(my_data2, centers = 8, nstart = 25)
  p8 <- fviz_cluster(k8, data = my_data2,repel = TRUE ,max.overlaps = Inf,labelsize = 8,main = "", palette = "jco",ggtheme = theme_bw())


  #Plotting the outputs
  #library(gridExtra)

  #Plot each individually
  #ggsave(file="whatever.pdf", p2) #saves  p2
  #ggsave(file="whatever.pdf", p3) #saves  p3
  #ggsave(file="whatever.pdf", p4) #saves  p4
  #ggsave(file="whatever.pdf", p5) #saves  p5
  #ggsave(file="whatever.pdf", p6) #saves  p6
  #ggsave(file="whatever.pdf", p7) #saves  p7
  #ggsave(file="whatever.pdf", p8) #saves  p8

  
  #plot1 <- arrangeGrob(p2, p3, p4, p5, nrow=2) #generates plot 1
  #ggsave(file="whatever.pdf", plot1) #saves plot 1

  #plot2 <- arrangeGrob(p6, p7, p8, nrow=2) #generates plot 2
  #ggsave(file="whatever.pdf", plot2) #saves plot 2

}

main()
