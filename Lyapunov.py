'''
This file plots the Lyapunov exponent that shows where chaos occurs for the
Logistic Map.

Note:

This is an early version of the Lyapunov Exponent

Comments are deliberately not cut out to allow for users to explore
different combinations.

This file is a transformation from an .ipynb -> .py

Author: Beverley-Claire Okogwu
'''

#!/usr/bin/env python
# coding: utf-8

# In[9]:


#import statements
import numpy as np
import matplotlib.pyplot as plt
import math
import random
get_ipython().run_line_magic('matplotlib', 'inline')

def tentMap(r, x_n):
    if x_n < 0.5:
        return r * x_n
    return r * (1.0-x_n)

# Generates a Lyapunov Exponent Plot
# The Lyapunov Exponent is time average of  log∣f′(xi)∣  at every state

def Lyapunov(start,end,step,x_n,freq):
    ret_lambdas =[]
    #define the r value range
    r_range = np.arange(start,end,step)

    #loop through each r value
    for rval in r_range:
        #array to hold the values
        hold =[]
        #generate a specific number of values based on the map
        for _ in range(freq):

            if x_n < 0.5:

                #replace with any chaotic map function
                x_n = tentMap(rval, x_n)

                # log∣f′(xi)∣=r

                hold.append(np.log(abs(rval)))
            else:
                x_n = tentMap(rval, x_n)

                # log∣f′(xi)∣=-1

                hold.append(math.log(abs(-1)))

        #get the average of the map values
        avg = np.mean(hold)
        #append the average to the lambda array
        ret_lambdas.append(avg)

    #PLOT THE LYAPUNOV EXPONENTS
    figure_plot = plt.figure(figsize=(10,10))
    axis = figure_plot.add_subplot(1,1,1)

    xticks = np.linspace(0, 2, 4000)
    # zero line
    zero = [0]*4000
    axis.plot(xticks, zero, 'g-')

    # plot lyapunov
    axis.plot(r_range, np.array(ret_lambdas), 'r-', linewidth = 3, label = 'Lyapunov exponent')
    axis.grid('on')
    axis.set_xlabel('r parameter values')
    axis.set_ylabel('Lyapunov')
    axis.legend(loc='best')
    axis.set_title('Tent Map versus Lyapunov exponent')


# In[10]:


Lyapunov(0.1,2.1,0.01,0.1,100)


# In[ ]:
