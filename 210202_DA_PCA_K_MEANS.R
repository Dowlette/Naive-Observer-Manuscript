install.packages("readxl")

library("readxl")

# Loading
#library("readxl")

# xlsx files
my_data <- read_excel(file.choose())
#you should only have the names in the columns, no row names
colnames(my_data) #identify the column names

library(tidyverse)
my_data2<-my_data %>% remove_rownames %>% column_to_rownames(var="names") #identify row names (no two row names can be the same)

#my_data <- read_excel("/Users/dowlettealameldin/Desktop/dhklab/StrokeProject/DatasetResults/DataSetZResults1.5.xlsx")

# Pricipal Components Analysis
# entering raw data and extracting PCs
# from the correlation matrix
#fit <- princomp(my_data, cor=TRUE)
# fit <- prcomp(my_data)
# summary(fit) # print variance accounted for
# loadings(fit) # pc loadings
# plot(fit,type="lines") # scree plot
# fit$scores # the principal components
# biplot(fit)

###%

# library(psych)
# fit <- principal(my_data, nfactors=4, rotate="varimax")
# fit # print results

###% THIS IS THE BEST ONE!!
# PCA Variable Factor Map
install.packages("FactoMineR")
library(FactoMineR)
result <- PCA(my_data2) # graphs generated automatically
#plot(result,repel = TRUE ,max.overlaps = Inf,main = "")
#write.csv(result, file="Users/dowlettealameldin/Desktop/dhklab/StrokeProject/datasetXPCAData.csv")
###% K-Means Clustering
#https://afit-r.github.io/kmeans_clustering this is where I got everything from 
install.packages("tidyverse")
install.packages("cluster")
install.packages("factoextra")
library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization
install.packages("ggrepel")
# k8 <- kmeans(my_data, centers = 4, nstart = 25)
# str(k8)
# #fviz_cluster(k8, geom = "point", ellipse.type = "norm"  )
# fviz_cluster(k8, data = my_data)

k2 <- kmeans(my_data2, centers = 2, nstart = 25)
#k3 <- kmeans(my_data2, centers = 3, nstart = 25)
#k4 <- kmeans(my_data2, centers = 4, nstart = 25)
#k5 <- kmeans(my_data2, centers = 5, nstart = 25)
#k6 <- kmeans(my_data2, centers = 6, nstart = 25)
k8 <- kmeans(my_data2, centers = 8, nstart = 25)

# # plots to compare
options(ggrepel.max.overlaps = Inf)
p2 <- fviz_cluster(k2, data = my_data2,repel = TRUE ,max.overlaps = Inf,labelsize = 8,main = "", palette = "jco", ggtheme = theme_bw())
#p3 <- fviz_cluster(k3, data = my_data2)
#p4 <- fviz_cluster(k4, data = my_data2)
#p5 <- fviz_cluster(k5, data = my_data2)
#p6 <- fviz_cluster(k6, data = my_data2)
p8 <- fviz_cluster(k8, data = my_data2,repel = TRUE ,max.overlaps = Inf,labelsize = 8,main = "", palette = "jco",ggtheme = theme_bw())

# 
library(gridExtra)
#grid.arrange(p2,p3,p4,p5, nrow = 2)
#grid.arrange(p4,p5, p6, p7, nrow = 2)
#grid.arrange(p6, p7, p8, p9, nrow = 2)
grid.arrange(p2, nrow = 1)
grid.arrange(p8, nrow = 1)

#library(tidyverse)  # data manipulation
#library(cluster)    # clustering algorithms
#library(factoextra) # clustering algorithms & visualization
set.seed(123)
fviz_nbclust(my_data2, kmeans, method = "wss")
fviz_nbclust(my_data2, kmeans, method = "silhouette")
