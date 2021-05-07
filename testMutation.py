'''
This file tests the mutate function.

Early version of the mutate function

Comments are deliberately not cut out to allow for users to explore
different combinations.

.ipynb -> .py

Author: Beverley-Claire Okogwu
'''
#!/usr/bin/env python
# coding: utf-8

# In[33]:


def logisticMap(r, x_n):
    return r * x_n *(1.0-x_n)


# In[40]:


import numpy as np
import matplotlib.pyplot as plt
import math
import random
#mutation function that will mutate an individual made up of floating point
#genes using a provided mutation probability (the probability of any gene being mutated)
#and a mutation function (e.g. Random, or one of the chaotic maps).

test= [0.8,0.9,8.8,-2.2,0.0]

# individual, the gene array
# mtype, "random" or "chaotic"
# cm_parameter, the r value of the chaotic map
# x_0, the initial value of the chaotic map
#probability, the probability of the gene being mutated (double value to 2dp e.g.0.25)
#map_type, LM,TM,YCM??
def mutation(individual,mtype,cm_parameter,x_0,probability):

    #for each individual's genes, get a random number between 0 and 1
    for gene in range(len(individual)):
        num = random.uniform(0.0,1.0)
        print("\nthe random number for {} is {}".format(individual[gene],num))
        #if this number <= the probability, then mutate
        if num <= probability:
            #if the type is random, then add a random number (from the Gaussian distribution)
            print("{} is less than or equal to {}. Let's mutate {}".format(num,probability,individual[gene]))
            if mtype=="random":
                print("Doing a random mutation....")
                value=np.random.normal()
                print("Adding {} to {}...".format(value,individual[gene]))
                individual[gene]+=value
                print("The value has been mutated to {}".format(individual[gene]))
            else:
                print("Doing a chaotic mutation....")
                #else, add a number from the chaotic map (default is logistic)
                value = logisticMap(cm_parameter,x_0)
                print("Adding {} to {}...".format(value,individual[gene]))
                individual[gene]+=value
                print("The value has been mutated to {}".format(individual[gene]))


    #return the mutated gene
    return individual



# In[44]:


print(mutation(test,"chaotic",3.8,0.1,0.25))


# In[ ]:
