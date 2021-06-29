#Y = read.xlsx("DataSetZResults1.5.xlsx",row.name=1) 
#Y=as.matrix(Y) 

# Install the released version from CRAN:
install.packages("corrgram")

# Install the development version from GitHub:
install.packages("devtools")
devtools::install_github("kwstat/corrgram")

install.packages("readxl")

library("readxl")

# Loading
#library("readxl")

# xlsx files
my_data4 <- read_excel(file.choose())
colnames(my_data4)

rownames(my_data4) <- c(colnames(my_data))
#my_data <- read_excel("/Users/dowlettealameldin/Desktop/dhklab/StrokeProject/DatasetResuls/DataSetZResults1.5.xlsx")

require(corrgram)
corrgram(my_data4, order=TRUE, lower.panel=panel.shade, upper.panel=panel.pie,
         text.panel=panel.txt, main="DataSetYCorrgram")
