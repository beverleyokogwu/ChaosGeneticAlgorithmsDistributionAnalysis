#!/usr/bin/env python
# coding: utf-8

# In[3]:


'''
Python Implementations of the Logistic, Tent, and Yang/Chen Chaotic Maps

Each function uses an initial x_0 and an r value to generate a specified number
of chaotic values.

Note: This file was originally converted from an ipynb.

Author: Beverley-Claire Okogwu

'''
#import statements
import numpy as np
import matplotlib.pyplot as plt
import math
import random
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


class ChaoticMaps():

    def __init__(self,r,x_n):
        self.r = r
        self.x_n = x_n

    # Logistic Map
    def logisticMap(self):
        return self.r * self.x_n *(1.0-self.x_n)


# In[5]:


# Tent Map
def tentMap(self):
    if self.x_n < 0.5:
        return self.r * self.x_n
    return self.r * (1.0-self.x_n)


# In[6]:


# Yang/Chen Map
def yangChen(self):
    return ((0.9*self.x_n)-(2*math.tanh(self.r * self.x_n)*math.exp(-3* math.pow(self.x_n,2))))


# In[ ]:
