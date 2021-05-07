
'''
This file shows the Shift-Scale function from thesis.

Comments are deliberately not cut out to allow for users to explore
different combinations.

Author: Beverley-Claire Okogwu
'''
#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
This function takes in a an array of chaotic map values
and shifts and/or scales the values to be in the range [-1,1]
'''
#import statements
import numpy as np
import matplotlib.pyplot as plt
import math
import random
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# a negative shift is a shift left,
# a positive shift is a shift right
def shift_scale(value_array,shift,scale):

    for val in range(len(value_array)):
        #do a shift and a scale
        value_array[val] = (value_array[val] +shift) * scale

    return value_array



# In[ ]:
