#!/usr/bin/env python
# coding: utf-8



'''
Python Implementations of the Griewank, Rosenbrock, and Rastrigin functions

Each function take in an array representing the "mutated values" from the
ShiftScale function (See ShiftScale.py).

Note: This file was originally converted from an ipynb.

Author: Beverley-Claire Okogwu

'''
#import statements
import numpy as np
import matplotlib.pyplot as plt
import math
import random
get_ipython().run_line_magic('matplotlib', 'inline')



class BenchMarkImplementations:

    def __init__(self, mutated_array):
        self.mutated_array = mutated_array


    # Griewank
    def griewank(self):

        #summation, sigma
        sigma=0
        #product, pi
        pi=1
        # length of the array
        d = len(self.mutated_array)

        for val in range(1,d):
            sigma+=math.pow(self.mutated_array[val],2)
            pi*=math.cos(float(self.mutated_array[val])/math.sqrt(val))

        return (float(sigma)/4000)-float(pi)+1


# Rosenbrock
def rosenbrock(self):
    #summation, sigma
    sigma=0
    # length of the array
    d = len(self.mutated_array)

    for elem in range(1,d-1):
        sigma+=(100*((self.mutated_array[elem+1]-(self.mutated_array[elem])**2)**2)+(self.mutated_array[elem]-1)**2)
    return sigma




# Rastrigin
def rastrigin(self):
    #summation, sigma
    sigma=0
    # length of the array
    d = len(self.mutated_array)

    for el in range(1,d):
        sigma+=((self.mutated_array[el]**2)-(10*math.cos(2*math.pi*self.mutated_array[el])))

    return (10*d)+sigma
