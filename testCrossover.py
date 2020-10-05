#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import matplotlib.pyplot as plt
import math
import random
#This function perform one point crossover with a random crossover point 
#for two individuals made up of floating point genes. 

#testgenes, size=5
i_1=[0.4, 0.5, 3.4, -8.8, 6.7]
i_2=[-1.5, 6.33, 9.99, 2.0, -0.03]

def crossover(indiv1,indiv2):
    
    # find the size of the genes (must be the same size, else cannot crossover)
    #assume gene sizes are the same for simplicity sake
    size = len(indiv1)
    
    #pick a random number in the range 0 to size-1 to denote the crosspoint(cp)
    crosspoint = random.randint(0,size-1)
    print("crosspoint : {}".format(crosspoint))
    
    #if crosspoint is 0, return indiv1
    if crosspoint==0:
        return indiv1
    #else, return a new array with the genes from 0 to the cp of indiv1 and from cp to the end of indiv2
    return indiv1[:crosspoint]+indiv2[crosspoint:]
    


# In[28]:


#test it out
print(crossover(i_1,i_2))


# In[ ]:





# In[ ]:




