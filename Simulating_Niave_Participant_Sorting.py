#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:42:07 2021

@author: dowlettealameldin
"""



# %%

#The matrix
#say images 1-15 so in python 0-14: are controls 
#say images 16-30 so in python 15-29: are GD
#say images 31-45 so in python 30-44: are OD
#say images 46-60 so in python 46-60: are OGD

#Initial Conditions -> this will change on the hypothesis we are testing
#Control images have 50% chance to be Control or GD
#GD images have 50% chance to be Control or GD
#OD images have 50% chance to be OD or OGD
#OGD images have 50% chance to be OD or OGD


# %%
import numpy as np
import random
#the random function makes it 50% automatically 
#use random.randint because it will give you a 50/50 chance of obtaining each number

#Make a list of control that is 105 in size becuase there are 15 images in this condition times 7 obervers = 105
Control=np.random.randint(1,3, size=105)
#make this list into an array
Control_Matrix=np.array(Control).reshape(15,7)

#Make a list of GD that is 105 in size becuase there are 15 images in this condition times 7 obervers = 105
GD=np.random.randint(1,3, size=105)
#make this list into an array
GD_Matrix=np.array(GD).reshape(15,7)

#Make a list of OD that is 105 in size becuase there are 15 images in this condition times 7 obervers = 105
OD=np.random.randint(3,5, size=105)
#make this list into an array
OD_Matrix=np.array(OD).reshape(15,7)

#Make a list of OGD that is 105 in size becuase there are 15 images in this condition times 7 obervers = 105
OGD=np.random.randint(3,5, size=105)
#make this list into an array
OGD_Matrix=np.array(OGD).reshape(15,7)

#Concatinate the Matrices
Naive_Particpant_Matrix=np.concatenate((Control_Matrix,GD_Matrix,OD_Matrix,OGD_Matrix))
Naive_Particpant_Matrix = pd.DataFrame(Naive_Particpant_Matrix)
Naive_Particpant_Matrix.to_excel('/Users/dowlettealameldin/Desktop/dhklab/StrokeProject/50_50_Simulated_Matrix.xlsx')


# %%
import numpy as np
import pandas as pd
import random

#the function makes it 75% of controls go to group 1 automatically 
Control=np.random.randint(1,5, size=105)
#make anything larger than 2 = 1 (were making the ration of group 1 to 2 = 1/4 -> 25%)
Control[Control > 2] = 1
Control_Matrix=np.array(Control).reshape(15,7)

#the function makes it 75% of GD go to group 2 automatically 
GD=np.random.randint(1,5, size=105)
GD[GD > 2] = 2
GD_Matrix=np.array(GD).reshape(15,7)


#the function makes it 75% of OD go to group 3 automatically 
OD=np.random.randint(3,7, size=105)
OD[OD > 4] = 3
OD_Matrix=np.array(OD).reshape(15,7)


#the function makes it 75% of OD go to group 4 automatically 
OGD=np.random.randint(3,7, size=105)
OGD[OGD > 4] = 4
OGD_Matrix=np.array(OGD).reshape(15,7)

Naive_Particpant_Matrix=np.concatenate((Control_Matrix,GD_Matrix,OD_Matrix,OGD_Matrix))

#Save Matrix to make into simularity matrix
Naive_Particpant_Matrix = np.array(Naive_Particpant_Matrix)
Naive_Particpant_Matrix = pd.DataFrame(Naive_Particpant_Matrix)
#Naive_Particpant_Matrix.to_excel('/Users/dowlettealameldin/Desktop/dhklab/StrokeProject/75_25_Simulated_Matrix.xlsx')


# %%
import numpy as np
import pandas as pd
import random

#the function makes it 83.333% of controls go to group 1 automatically 
Control=np.random.randint(1,7, size=105)
#make anything larger than 2 = 1 (were making the ration of group 1 to 2 = 1/4 -> 25%)
Control[Control > 2] = 1
Control_Matrix=np.array(Control).reshape(15,7)

#the function makes it 83.333% of GD go to group 2 automatically 
GD=np.random.randint(1,7, size=105)
GD[GD > 2] = 2
GD_Matrix=np.array(GD).reshape(15,7)


#the function makes it 83.333% of OD go to group 3 automatically 
OD=np.random.randint(3,9, size=105)
OD[OD > 4] = 3
OD_Matrix=np.array(OD).reshape(15,7)


#the function makes it 83.333% of OD go to group 4 automatically 
OGD=np.random.randint(3,9, size=105)
OGD[OGD > 4] = 4
OGD_Matrix=np.array(OGD).reshape(15,7)

Naive_Particpant_Matrix=np.concatenate((Control_Matrix,GD_Matrix,OD_Matrix,OGD_Matrix))

#Save Matrix to make into simularity matrix
Naive_Particpant_Matrix = np.array(Naive_Particpant_Matrix)
Naive_Particpant_Matrix = pd.DataFrame(Naive_Particpant_Matrix)
Naive_Particpant_Matrix.to_excel('/Users/dowlettealameldin/Desktop/dhklab/StrokeProject/83.3_16.6_Simulated_Matrix.xlsx')


# %%
import numpy as np
import pandas as pd
import random

#the function makes it 90% of controls go to group 1 automatically 
Control=np.random.randint(1,11, size=105)
#make anything larger than 2 = 1 (were making the ration of group 1 to 2 = 1/4 -> 25%)
Control[Control > 2] = 1
Control_Matrix=np.array(Control).reshape(15,7)

#the function makes it 90% of GD go to group 2 automatically 
GD=np.random.randint(1,11, size=105)
GD[GD > 2] = 2
GD_Matrix=np.array(GD).reshape(15,7)


#the function makes it 90% of OD go to group 3 automatically 
OD=np.random.randint(3,13, size=105)
OD[OD > 4] = 3
OD_Matrix=np.array(OD).reshape(15,7)


#the function makes it 90% of OD go to group 4 automatically 
OGD=np.random.randint(3,13, size=105)
OGD[OGD > 4] = 4
OGD_Matrix=np.array(OGD).reshape(15,7)

Naive_Particpant_Matrix=np.concatenate((Control_Matrix,GD_Matrix,OD_Matrix,OGD_Matrix))

#Save Matrix to make into simularity matrix
Naive_Particpant_Matrix = np.array(Naive_Particpant_Matrix)
Naive_Particpant_Matrix = pd.DataFrame(Naive_Particpant_Matrix)
Naive_Particpant_Matrix.to_excel('/Users/dowlettealameldin/Desktop/dhklab/StrokeProject/90_10_Simulated_Matrix.xlsx')


# %%
import numpy as np
import pandas as pd
import random

#the function makes it 95% of controls go to group 1 automatically 
Control=np.random.randint(1,21, size=105)
#make anything larger than 2 = 1 (were making the ration of group 1 to 2 = 1/4 -> 25%)
Control[Control > 2] = 1
Control_Matrix=np.array(Control).reshape(15,7)

#the function makes it 95% of GD go to group 2 automatically 
GD=np.random.randint(1,21, size=105)
GD[GD > 2] = 2
GD_Matrix=np.array(GD).reshape(15,7)


#the function makes it 95% of OD go to group 3 automatically 
OD=np.random.randint(3,23, size=105)
OD[OD > 4] = 3
OD_Matrix=np.array(OD).reshape(15,7)


#the function makes it 95% of OD go to group 4 automatically 
OGD=np.random.randint(3,23, size=105)
OGD[OGD > 4] = 4
OGD_Matrix=np.array(OGD).reshape(15,7)

Naive_Particpant_Matrix=np.concatenate((Control_Matrix,GD_Matrix,OD_Matrix,OGD_Matrix))

#Save Matrix to make into simularity matrix
Naive_Particpant_Matrix = np.array(Naive_Particpant_Matrix)
Naive_Particpant_Matrix = pd.DataFrame(Naive_Particpant_Matrix)
Naive_Particpant_Matrix.to_excel('/Users/dowlettealameldin/Desktop/dhklab/StrokeProject/95_5_Simulated_Matrix.xlsx')


# %%
import numpy as np
import pandas as pd
import random

#the function makes it 98% of controls go to group 1 automatically 
Control=np.random.randint(1,51, size=105)
#make anything larger than 2 = 1 (were making the ration of group 1 to 2 = 1/4 -> 25%)
Control[Control > 2] = 1
Control_Matrix=np.array(Control).reshape(15,7)

#the function makes it 98% of GD go to group 2 automatically 
GD=np.random.randint(1,51, size=105)
GD[GD > 2] = 2
GD_Matrix=np.array(GD).reshape(15,7)


#the function makes it 98% of OD go to group 3 automatically 
OD=np.random.randint(3,53, size=105)
OD[OD > 4] = 3
OD_Matrix=np.array(OD).reshape(15,7)


#the function makes it 98% of OD go to group 4 automatically 
OGD=np.random.randint(3,53, size=105)
OGD[OGD > 4] = 4
OGD_Matrix=np.array(OGD).reshape(15,7)

Naive_Particpant_Matrix=np.concatenate((Control_Matrix,GD_Matrix,OD_Matrix,OGD_Matrix))

#Save Matrix to make into simularity matrix
Naive_Particpant_Matrix = np.array(Naive_Particpant_Matrix)
Naive_Particpant_Matrix = pd.DataFrame(Naive_Particpant_Matrix)
Naive_Particpant_Matrix.to_excel('/Users/dowlettealameldin/Desktop/dhklab/StrokeProject/98_2_Simulated_Matrix.xlsx')


