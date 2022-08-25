install.packages("readxl")

library("readxl")

#Load the xlsx file
#You should use simularity array or corrgram as datafile input (it will not change the PCA or K-Means output
my_data <- read_excel(file.choose())
#my_data <- read_excel("/Users/dowlettealameldin/Desktop/dhklab/StrokeProject/DatasetResults/DataSetZResults1.5.xlsx")
#you should only have the names in the columns, no row names
colnames(my_data) #identify the column names

library(tidyverse)
my_data2<-my_data %>% remove_rownames %>% column_to_rownames(var="names") #identify row names (no two row names can be the same)


# PCA Variable Factor Map
install.packages("FactoMineR")
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

#If your optimal number of clusters is 8, run the section below
k8 <- kmeans(my_data2, centers = 8, nstart = 25)
p8 <- fviz_cluster(k8, data = my_data2,repel = TRUE ,max.overlaps = Inf,labelsize = 8,main = "", palette = "jco",ggtheme = theme_bw())


#Plotting the outputs
library(gridExtra)

#Plot each individually
grid.arrange(p2, nrow = 1)
grid.arrange(p3, nrow = 1)
grid.arrange(p4, nrow = 1)
grid.arrange(p5, nrow = 1)
grid.arrange(p6, nrow = 1)
grid.arrange(p7, nrow = 1)
grid.arrange(p8, nrow = 1)

#Plot the outputs together 
#k=2,k=3,k=4,k=5 together
grid.arrange(p2,p3,p4,p5, nrow = 2)
#k=4,k=5,k=6,k=7 together
grid.arrange(p4,p5, p6, p7, nrow = 2)
#k=6,k=7,k=8 together
grid.arrange(p6, p7, p8, nrow = 2)




